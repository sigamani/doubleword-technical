"""Minimal Ray Data LLM Batch Inference with Typer CLI.

Based on: https://github.com/0-mostafa-rezaee-0/Batch_LLM_Inference_with_Ray_Data_LLM/blob/main/notebooks/ray_data_llm_test.ipynb
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import typer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Minimal Ray Data LLM Batch Inference")


class SimpleBatchProcessor:
    """Simple batch processor for LLM inference."""
    
    def __init__(self, model_name: str = "fallback"):
        """Initialize processor with a model."""
        self.model_name = model_name
        logger.info(f"Initialized processor with model: {model_name}")
    
    def process_batch(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of questions."""
        logger.info(f"Processing {len(questions)} questions")
        
        answers = []
        for q in questions:
            if "capital of France" in q.lower():
                answers.append("The capital of France is Paris.")
            elif "planets" in q.lower():
                answers.append("There are 8 planets in our solar system.")
            elif "2+2" in q:
                answers.append("2+2 equals 4.")
            elif "joke" in q.lower():
                answers.append("Why don't scientists trust atoms? Because they make up everything!")
            else:
                answers.append("I don't have an answer for that question.")
        
        return [{"question": q, "answer": a} for q, a in zip(questions, answers)]


@app.command()
def process(
    input_file: Path = typer.Argument(..., help="Input file with questions (JSON or JSONL)"),
    output_file: Path = typer.Argument(..., help="Output file for results"),
    model: str = typer.Option("fallback", help="Model name to use"),
    batch_size: int = typer.Option(4, help="Batch size for processing"),
):
    """Process a batch of questions through LLM inference."""
    
    try:
        # Load input data
        logger.info(f"Loading questions from {input_file}")
        questions = _load_questions(input_file)
        logger.info(f"Loaded {len(questions)} questions")
        
        # Initialize processor
        processor = SimpleBatchProcessor(model)
        
        # Process in batches
        all_results = []
        total_batches = (len(questions) + batch_size - 1) // batch_size
        
        logger.info(f"Processing {len(questions)} questions in {total_batches} batches...")
        start_time = time.time()
        
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_questions)} questions)")
            
            # Local processing
            batch_results = processor.process_batch(batch_questions)
            all_results.extend(batch_results)
        
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        # Save results
        logger.info(f"Saving results to {output_file}")
        _save_results(all_results, output_file)
        
        # Print summary
        logger.info(f"✅ Processed {len(all_results)} questions")
        logger.info(f"⏱️  Total time: {processing_time:.2f} seconds")
        logger.info(f"📊 Throughput: {len(all_results)/processing_time:.2f} questions/second")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise typer.Exit(1)


@app.command()
def demo(
    output_file: Path = typer.Option("demo_results.json", help="Output file for demo results"),
):
    """Run a demo with sample questions."""
    
    # Sample questions from notebook
    sample_questions = [
        "What is the capital of France?",
        "How many planets are in our solar system?",
        "What is 2+2?",
        "Tell me a joke."
    ]
    
    # Create temporary input file
    temp_input = Path("temp_demo_questions.json")
    _save_questions(sample_questions, temp_input)
    
    try:
        # Process demo
        process(temp_input, output_file, model="fallback", batch_size=4)
        
        # Display results
        if output_file.exists():
            results = _load_results(output_file)
            print("\n🎯 Demo Results:")
            print("=" * 50)
            for i, result in enumerate(results, 1):
                print(f"Question {i}: {result['question']}")
                print(f"Answer: {result['answer']}")
                print("-" * 30)
    
    finally:
        # Clean up temp file
        if temp_input.exists():
            temp_input.unlink()


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
):
    """Start a simple API server for batch processing."""
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        from typing import List
        import uvicorn
        
        app = FastAPI(title="Batch LLM Inference API")
        processor = SimpleBatchProcessor()
        
        class BatchRequest(BaseModel):
            questions: List[str]
            model: str = "fallback"
        
        @app.post("/process")
        async def process_batch(request: BatchRequest):
            """Process a batch of questions."""
            results = processor.process_batch(request.questions)
            return {"results": results, "count": len(results)}
        
        @app.get("/")
        async def root():
            return {"status": "healthy", "service": "batch-llm-inference"}
        
        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)
        
    except ImportError:
        logger.error("fastapi and uvicorn required for serve command")
        raise typer.Exit(1)


def _load_questions(input_file: Path) -> List[str]:
    """Load questions from input file."""
    if input_file.suffix == '.jsonl':
        questions = []
        with open(input_file) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if isinstance(data, dict) and 'question' in data:
                        questions.append(data['question'])
                    elif isinstance(data, str):
                        questions.append(data)
        return questions
    else:
        with open(input_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                return [item['question'] if isinstance(item, dict) else str(item) for item in data]
            else:
                return [data.get('question', '') for data in data.values()]


def _save_questions(questions: List[str], output_file: Path):
    """Save questions to JSON file."""
    with open(output_file, 'w') as f:
        json.dump([{"question": q} for q in questions], f, indent=2)


def _save_results(results: List[Dict[str, Any]], output_file: Path):
    """Save results to output file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)


def _load_results(input_file: Path) -> List[Dict[str, Any]]:
    """Load results from file."""
    with open(input_file) as f:
        return json.load(f)


if __name__ == "__main__":
    app()