import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from api.routes import app


if __name__ == "__main__":
    print("Starting Ray Data vLLM Batch Inference Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)