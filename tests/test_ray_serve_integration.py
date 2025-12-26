import time
import os
import sys
import pytest
import threading
from pathlib import Path
from typing import List, Tuple, Optional

import requests
import concurrent.futures
import ray
from ray import serve

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from test_ray_serve import VLLMDeployment, GenerationResult


class TestRayServeIntegration:

    @pytest.fixture(scope="class")
    def ray_serve_setup(self):
        """Setup Ray Serve with vLLM deployment for test suite."""
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        ray.init(num_cpus=4, ignore_reinit_error=True)
        serve.start(http_options={"host": "127.0.0.1", "port": 8000})
        serve.run(VLLMDeployment.bind())
        time.sleep(20)

        yield

        serve.shutdown()
        ray.shutdown()

    @pytest.fixture
    def test_prompts(self) -> List[str]:
        return [
            "What is capital of France?",
            "Tell me a joke about programming.",
            "Explain machine learning in simple terms.",
            "What is meaning of life?",
        ]

    def test_deployment_initialization(self):
        status = serve.status()
        assert status.applications["default"].status == "RUNNING"

    def test_generation_result_structure(self):
        result = GenerationResult(
            prompt="test prompt",
            request_id="test_id",
            response="test response",
            tokens=10,
            worker_id="test_worker",
            generation_time=0.5,
            finish_reason="length",
            error=None,
        )

        assert result.prompt == "test prompt"
        assert result.request_id == "test_id"
        assert result.response == "test response"
        assert result.tokens == 10
        assert result.worker_id == "test_worker"
        assert result.generation_time == 0.5
        assert result.finish_reason == "length"
        assert result.error is None

    def test_http_endpoint_availability(self):
        response = requests.post(
            "http://127.0.0.1:8000/",
            json={"text": "Hello", "request_id": "test"},
            timeout=30,
        )

        assert response.status_code == 200

        result_dict = response.json()
        required_fields = [
            "prompt",
            "response",
            "worker_id",
            "generation_time",
            "tokens",
        ]
        assert all(field in result_dict for field in required_fields)

    def test_concurrent_requests(self, test_prompts):
        results = self._run_requests_concurrently(test_prompts)
        valid_results = [r for r in results if r[0] is not None]

        assert len(valid_results) == len(test_prompts)

        for result, elapsed in valid_results:
            assert elapsed < 60
            assert result.generation_time > 0
            assert result.response is not None
            assert result.tokens > 0

    def test_worker_distribution(self, test_prompts):
        worker_ids = self._fetch_worker_ids_concurrently(test_prompts)

        assert len(worker_ids) == len(test_prompts)
        assert all(worker_id is not None for worker_id in worker_ids)

    def test_error_handling(self):
        response = requests.post("http://127.0.0.1:8000/", json={}, timeout=30)
        assert response.status_code == 200

        response = requests.post(
            "http://127.0.0.1:8000/",
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        assert response.status_code in [400, 500]

    def test_original_workflow(self, test_prompts):
        """Test the original workflow from main() function."""
        start_total = time.time()
        results = self._run_requests_concurrently(test_prompts)
        self._assert_valid_results(results, test_prompts, time.time() - start_total)

    def _run_http_request(
        self, prompt: str, idx: int
    ) -> Tuple[Optional[GenerationResult], float]:
        start = time.time()
        response = requests.post(
            "http://127.0.0.1:8000/",
            json={"text": prompt, "request_id": idx},
            timeout=30,
        )
        elapsed = time.time() - start

        if response.status_code == 200:
            result_dict = response.json()
            return GenerationResult(**result_dict), elapsed
        return None, elapsed

    def _run_requests_concurrently(
        self, prompts: List[str]
    ) -> List[Tuple[Optional[GenerationResult], float]]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self._run_http_request, prompt, i)
                for i, prompt in enumerate(prompts)
            ]
            return [f.result() for f in concurrent.futures.as_completed(futures)]

    def _fetch_worker_ids_concurrently(self, prompts: List[str]) -> List[Optional[str]]:
        results = self._run_requests_concurrently(prompts)
        return [result.worker_id if result else None for result, _ in results]

    def _assert_valid_results(
        self,
        results: List[Tuple[Optional[GenerationResult], float]],
        expected_prompts: List[str],
        total_time: float,
    ) -> None:
        valid_results = [r for r in results if r[0] is not None]
        assert len(valid_results) == len(expected_prompts)
        assert valid_results

        if valid_results:
            avg_latency = sum(r[1] for r in valid_results) / len(valid_results)
            assert avg_latency > 0
            assert total_time > 0

    def test_shutdown_signal(self):
        shutdown_event = threading.Event()

        def simulate_main_loop():
            try:
                while not shutdown_event.is_set():
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass

        loop_thread = threading.Thread(target=simulate_main_loop)
        loop_thread.start()
        time.sleep(0.5)
        shutdown_event.set()
        loop_thread.join(timeout=1)

        assert not loop_thread.is_alive()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
