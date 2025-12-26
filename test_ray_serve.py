import time
import os
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import ray
from ray import serve
import requests
import concurrent.futures

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = ""


@dataclass
class GenerationResult:
    prompt: str
    request_id: Optional[str]
    response: Optional[str]
    tokens: int
    worker_id: str
    generation_time: float
    finish_reason: Optional[str] = None
    error: Optional[str] = None


async def parse_request_body(request) -> Dict:
    if hasattr(request, "body"):
        body = await request.body()
        body = body.decode("utf-8") if isinstance(body, bytes) else body
        return json.loads(body)
    return request if isinstance(request, dict) else {"text": str(request)}


def create_success_result(
    prompt: str, request_id: Optional[str], output, worker_id: str, elapsed: float
) -> GenerationResult:
    completion_output = output.outputs[0]
    generated_text = completion_output.text
    token_ids = completion_output.token_ids
    finish_reason = completion_output.finish_reason

    return GenerationResult(
        prompt=prompt,
        request_id=request_id,
        response=generated_text,
        tokens=len(token_ids),
        worker_id=worker_id,
        generation_time=elapsed,
        finish_reason=finish_reason,
    )


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 2, "num_gpus": 0})
class VLLMDeployment:
    def __init__(self):
        from vllm import LLM, SamplingParams

        worker_id = ray.get_runtime_context().get_worker_id()
        logger.info(f"[Worker {worker_id}] Initializing vLLM on CPU...")
        self.llm = LLM(
            model="facebook/opt-125m", enforce_eager=True, tensor_parallel_size=1
        )
        self.sampling_params = SamplingParams(temperature=0.7, max_tokens=50, top_p=0.9)
        logger.info(f"[Worker {worker_id}] vLLM initialized successfully")

    async def __call__(self, request):
        data = await parse_request_body(request)
        prompt = data.get("text", "")
        request_id = data.get("request_id")
        return self._generate_response(prompt, request_id)

    def _generate_response(
        self, prompt: str, request_id: Optional[str]
    ) -> GenerationResult:
        start_time = time.time()
        worker_id = ray.get_runtime_context().get_worker_id()

        try:
            outputs = self.llm.generate([prompt], self.sampling_params)
            output = outputs[0]
            elapsed = time.time() - start_time
            return create_success_result(prompt, request_id, output, worker_id, elapsed)
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Worker {worker_id}] Generation error: {str(e)}")
            return GenerationResult(
                prompt=prompt,
                request_id=request_id,
                response=None,
                tokens=0,
                worker_id=worker_id,
                generation_time=elapsed,
                error=str(e),
            )


def send_http_request(
    prompt: str, idx: int
) -> Tuple[Optional[GenerationResult], float]:
    start = time.time()
    response = requests.post(
        "http://127.0.0.1:8000/", json={"text": prompt, "request_id": idx}, timeout=30
    )
    elapsed = time.time() - start
    return _parse_http_response(response, elapsed)


def _parse_http_response(
    response, elapsed: float
) -> Tuple[Optional[GenerationResult], float]:
    if response.status_code == 200:
        result_dict = response.json()
        result = GenerationResult(**result_dict)
        return result, elapsed
    logger.error(f"Request failed with status {response.status_code}: {response.text}")
    return None, elapsed


def run_requests_concurrently(
    prompts: List[str],
) -> List[Tuple[Optional[GenerationResult], float]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(send_http_request, p, i) for i, p in enumerate(prompts)
        ]
        return [f.result() for f in concurrent.futures.as_completed(futures)]


def calculate_avg_latency(
    results: List[Tuple[Optional[GenerationResult], float]],
) -> float:
    return sum(r[1] for r in results) / len(results)


def log_summary(
    results: List[Tuple[Optional[GenerationResult], float]], total_time: float):
    valid_results = [r for r in results if r[0] is not None]
    logger.info(f"Completed {len(valid_results)} requests in {total_time:.2f}s")
    if valid_results:
        logger.info(
            f"Average request latency: {calculate_avg_latency(valid_results):.2f}s"
        )


def run_test_requests(prompts: List[str]):
    logger.info("Testing distributed vLLM workers via HTTP.")
    start_total = time.time()
    results = run_requests_concurrently(prompts)
    log_summary(results, time.time() - start_total)


