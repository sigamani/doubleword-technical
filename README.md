# Minimal Ray Data LLM Batch Inference

A clean, minimal implementation of batch LLM inference using Typer CLI, based on the Ray Data LLM workflow.

## Features

- **Unified CLI Interface**: Single command-line tool for all operations
- **Batch Processing**: Process multiple questions efficiently in batches
- **Flexible Input/Output**: Support for JSON and JSONL formats
- **Demo Mode**: Built-in demo with sample questions
- **API Server**: Simple FastAPI server for batch processing
- **Fallback Logic**: Graceful degradation when models aren't available

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Process Questions from File

```bash
python main.py process sample_questions.json results.json
```

### Run Demo

```bash
python main.py demo
```

### Start API Server

```bash
python main.py serve --host 0.0.0.0 --port 8000
```

### Get Help

```bash
python main.py --help
python main.py process --help
```

## Commands

### `process`
Process a batch of questions through LLM inference.

**Arguments:**
- `input_file`: Input file with questions (JSON or JSONL)
- `output_file`: Output file for results

**Options:**
- `--model`: Model name to use (default: fallback)
- `--batch-size`: Batch size for processing (default: 4)

### `demo`
Run a demo with sample questions.

**Options:**
- `--output-file`: Output file for demo results (default: demo_results.json)

### `serve`
Start a simple API server for batch processing.

**Options:**
- `--host`: Host to bind to (default: 127.0.0.1)
- `--port`: Port to bind to (default: 8000)

## Input Formats

### JSON Format
```json
[
  {"question": "What is the capital of France?"},
  {"question": "How many planets are in our solar system?"}
]
```

### JSONL Format
```jsonl
{"question": "What is the capital of France?"}
{"question": "How many planets are in our solar system?"}
```

## Output Format

```json
[
  {
    "question": "What is the capital of France?",
    "answer": "The capital of France is Paris."
  },
  {
    "question": "How many planets are in our solar system?",
    "answer": "There are 8 planets in our solar system."
  }
]
```

## API Endpoints

When running the server:

### `POST /process`
Process a batch of questions.

**Request:**
```json
{
  "questions": ["What is 2+2?", "Tell me a joke"],
  "model": "fallback"
}
```

**Response:**
```json
{
  "results": [
    {"question": "What is 2+2?", "answer": "2+2 equals 4."},
    {"question": "Tell me a joke", "answer": "Why don't scientists trust atoms?..."}
  ],
  "count": 2
}
```

### `GET /`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "batch-llm-inference"
}
```

## Architecture

This implementation follows the core workflow from the original Ray Data LLM notebook:

1. **Input Loading**: Load questions from JSON/JSONL files
2. **Batch Processing**: Process questions in configurable batch sizes
3. **Model Interface**: Clean abstraction for different model backends
4. **Fallback Logic**: Graceful handling when models aren't available
5. **Output Generation**: Save results in structured JSON format

## Future Enhancements

- Ray integration for distributed processing
- Real LLM model integration (TinyLlama, etc.)
- GPU acceleration support
- Advanced batching strategies
- Performance monitoring
- Configuration management

## Development

This is a clean, minimal implementation designed to demonstrate:

- ✅ Clean Typer CLI interface
- ✅ Proper error handling
- ✅ Flexible input/output formats
- ✅ Batch processing logic
- ✅ API server integration
- ✅ Fallback mechanisms

Perfect starting point for building a production-ready batch inference system.