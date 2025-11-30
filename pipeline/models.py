""" Data models for batch processing pipeline. """

from dataclasses import dataclass
from typing import Any

@dataclass
class InferenceRequest:
    prompt: str


@dataclass
class InferenceResult:
    prompt: str
    response: str
    tokens: int
    processing_time: float
    
    def to_dict(self):
        return {
            "prompt": self.prompt,
            "response": self.response,
            "tokens": self.tokens,
            "processing_time": self.processing_time
        }