#!/usr/bin/env python3
"""
Client script for sending batch requests to LLM inference server
Run this on your MacBook Pro
"""

import requests
import time
from typing import List, Dict
import argparse


class LLMClient:
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")
        self.session = requests.Session()

    def health_check(self) -> Dict:
        """Check server health"""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def generate_text(
        self, prompt: str, max_tokens: int = 100, temperature: float = 0.7
    ) -> Dict:
        """Generate single text"""
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            response = self.session.post(
                f"{self.server_url}/generate", json=payload, timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def generate_batch(
        self, prompts: List[str], max_tokens: int = 100, temperature: float = 0.7
    ) -> Dict:
        """Generate batch text"""
        payload = {
            "prompts": prompts,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            response = self.session.post(
                f"{self.server_url}/generate_batch",
                json=payload,
                timeout=300,  # 5 minutes for batch
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="LLM Inference Client")
    parser.add_argument(
        "--server", required=True, help="Server URL (e.g., http://192.168.1.100:8000)"
    )
    parser.add_argument("--prompt", help="Single prompt to process")
    parser.add_argument("--batch", action="store_true", help="Process batch prompts")
    parser.add_argument("--file", help="File containing prompts (one per line)")
    parser.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Generation temperature"
    )
    parser.add_argument("--health", action="store_true", help="Check server health")

    args = parser.parse_args()

    # Initialize client
    client = LLMClient(args.server)

    # Health check
    if args.health:
        print("Checking server health...")
        health = client.health_check()
        if "error" in health:
            print(f"Health check failed: {health['error']}")
            return
        print("Server is healthy:")
        print(f"   Worker: {health.get('worker_id', 'Unknown')}")
        print(f"   Model: {health.get('model', 'Unknown')}")
        print(f"   Device: {health.get('device', 'Unknown')}")
        print(
            f"   GPU Memory: {health.get('gpu_memory_used_gb', 'Unknown')}GB / {health.get('gpu_memory_total_gb', 'Unknown')}GB"
        )
        print(f"   Queue Size: {health.get('queue_size', 'Unknown')}")
        return

    # Single prompt
    if args.prompt:
        print("Processing single prompt...")
        print(f"Prompt: {args.prompt[:100]}...")

        start_time = time.time()
        result = client.generate_text(args.prompt, args.max_tokens, args.temperature)
        total_time = time.time() - start_time

        if "error" in result:
            print(f"Generation failed: {result['error']}")
            return

        print("Generation completed:")
        print(f"   Text: {result['text']}")
        print(f"   Tokens: {result['tokens_generated']}")
        print(f"   Inference Time: {result['inference_time']:.2f}s")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Worker: {result['worker_id']}")
        return

    # Batch processing
    prompts = []

    if args.file:
        print(f"Loading prompts from {args.file}...")
        try:
            with open(args.file, "r") as f:
                prompts = [line.strip() for line in f if line.strip()]
            print(f"   Loaded {len(prompts)} prompts")
        except Exception as e:
            print(f"Failed to load file: {e}")
            return
    else:
        # Default batch prompts
        prompts = [
            "What is the future of artificial intelligence?",
            "Explain quantum computing in simple terms.",
            "How does machine learning work?",
            "What are the applications of deep learning?",
            "Describe the impact of AI on society.",
            "What is the difference between AI and machine learning?",
            "How do neural networks learn?",
            "What is natural language processing?",
            "Explain computer vision applications.",
            "What are the ethical considerations in AI development?",
        ]
        print(f"Using {len(prompts)} default prompts")

    print(f"Processing batch of {len(prompts)} prompts...")
    print(f"   Max Tokens: {args.max_tokens}")
    print(f"   Temperature: {args.temperature}")

    start_time = time.time()
    result = client.generate_batch(prompts, args.max_tokens, args.temperature)
    total_time = time.time() - start_time

    if "error" in result:
        print(f"Batch generation failed: {result['error']}")
        return

    print(f"Batch completed in {total_time:.2f}s")
    print(f"   Worker: {result['worker_id']}")
    print(f"   Total Time: {result['total_time']:.2f}s")
    print(
        f"   Average per prompt: {result['total_time'] / len(result['results']):.2f}s"
    )

    total_tokens = 0
    total_inference_time = 0

    print("\nResults:")
    for i, res in enumerate(result["results"]):
        total_tokens += res["tokens_generated"]
        total_inference_time += res["inference_time"]
        print(
            f"   {i + 1:2d}. Tokens: {res['tokens_generated']:3d} | Time: {res['inference_time']:5.2f}s | {res['text'][:60]}..."
        )

    print("\nSummary:")
    print(f"   Total Tokens Generated: {total_tokens}")
    print(f"   Total Inference Time: {total_inference_time:.2f}s")
    print(f"   Average Tokens/Second: {total_tokens / total_inference_time:.1f}")
    print(
        f"   Throughput: {len(result['results']) / result['total_time']:.1f} prompts/second"
    )


if __name__ == "__main__":
    main()
