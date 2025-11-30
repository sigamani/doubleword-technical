import ray
import pandas as pd
import logging
from typing import List

logger = logging.getLogger(__name__)

def create_dataset(prompts: List[str]):
    """Create a Ray Dataset from prompts"""
    try:
        # Create dataset from list of prompts
        data = [{"prompt": prompt} for prompt in prompts]
        df = pd.DataFrame(data)
        ds = ray.data.from_pandas(df)
        logger.info(f"Created Ray dataset with {len(prompts)} prompts")
        return ds
    except Exception as e:
        logger.error(f"Failed to create Ray dataset: {e}")
        # Fallback to simple list
        return prompts