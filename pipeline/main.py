""" Main pipeline for batch processing using Ray. """

import logging
from ..config import EnvironmentConfig, ModelConfig
from .ray_batch import RayBatchProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_prompts() -> list:
    return [
        "Hello world",
        "What is artificial intelligence?",
        "Explain machine learning",
        "Test prompt 4",
        "Test prompt 5"
    ]


def log_results(results: list):
    logger.info(f"Processed {len(results)} prompts")
    for i, result in enumerate(results):
        if isinstance(result, dict):
            _log_single_result(i, result)


def _log_single_result(index: int, result: dict):
    prompt = result.get('prompt', str(result))[:30]
    response = result.get('response', str(result))[:30]
    logger.info(f"{index+1}. {prompt}... -> {response}...")


def main():
    env_config = EnvironmentConfig.from_env()
    model_config = ModelConfig.default()
    processor = RayBatchProcessor(model_config, env_config)
    
    prompts = create_test_prompts()
    results = processor.process_batch(prompts)
    log_results(results)


if __name__ == "__main__":
    main()