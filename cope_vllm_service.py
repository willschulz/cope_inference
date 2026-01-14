#!/usr/bin/env python3
"""
CoPE-A-9B Inference Service using vLLM

High-throughput inference using vLLM's continuous batching.
Significantly faster than the standard transformers-based service.
"""

import logging
import time
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = os.path.expanduser("~/cope_inference/cope-merged")

# Global model state
llm = None


def build_prompt(policy: str, content: str) -> str:
    """
    Build the CoPE prompt from policy and content.
    """
    prompt = f"""INSTRUCTIONS
============

Examine the given POLICY and determine if the given CONTENT meets the criteria for ANY of the LABELS. Answer "1" if yes, and "0" if no.


POLICY
======

{policy}


CONTENT
=======

{content}


ANSWER
======

"""
    return prompt


def load_model():
    """
    Load the merged CoPE model with vLLM.
    """
    global llm
    
    logger.info(f"Loading model with vLLM: {MODEL_PATH}")
    logger.info("Using bfloat16 with vLLM optimizations...")
    
    # vLLM automatically handles GPU memory efficiently
    # Gemma2 requires bfloat16 for numerical stability
    # Note: BF16 model is ~17GB, so we limit context to leave room for KV cache
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=0.95,  # Use 95% of GPU memory
        max_model_len=2048,  # Sufficient for policy + tweet (typically <1K tokens)
        trust_remote_code=True,
    )
    
    logger.info("Model loaded successfully with vLLM!")
    return llm


def parse_label(output_text: str) -> int:
    """Parse model output to binary label."""
    output_text = output_text.strip()
    if output_text.startswith("1"):
        return 1
    elif output_text.startswith("0"):
        return 0
    elif "1" in output_text:
        return 1
    elif "0" in output_text:
        return 0
    else:
        logger.warning(f"Unexpected model output: '{output_text}', defaulting to 0")
        return 0


# Sampling params for classification (greedy, single token)
SAMPLING_PARAMS = SamplingParams(
    temperature=0,
    max_tokens=1,
)


# Pydantic models for API
class LabelRequest(BaseModel):
    policy: str
    content: str


class LabelResponse(BaseModel):
    label: int
    raw_output: str


class BatchItem(BaseModel):
    id: str = Field(..., description="Unique identifier for this item")
    content: str = Field(..., description="Text content to classify")


class BatchRequest(BaseModel):
    policy: str = Field(..., description="The content policy to apply to all items")
    items: List[BatchItem] = Field(..., description="List of items to classify")


class BatchResultItem(BaseModel):
    id: str
    label: int
    raw_output: str


class BatchResponse(BaseModel):
    results: List[BatchResultItem]
    processed: int
    elapsed_seconds: float
    items_per_second: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_allocated_gb: Optional[float] = None
    backend: str = "vllm"


# FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    logger.info("Starting CoPE-A-9B vLLM inference service...")
    load_model()
    yield
    logger.info("Shutting down CoPE-A-9B vLLM inference service...")


app = FastAPI(
    title="CoPE-A-9B Inference Service (vLLM)",
    description="High-throughput content policy evaluation using vLLM",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check service health and model status."""
    gpu_name = None
    gpu_memory = None
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = round(torch.cuda.memory_allocated() / 1024**3, 2)
    
    return HealthResponse(
        status="healthy",
        model_loaded=llm is not None,
        gpu_available=torch.cuda.is_available(),
        gpu_name=gpu_name,
        gpu_memory_allocated_gb=gpu_memory,
        backend="vllm",
    )


@app.post("/label", response_model=LabelResponse)
async def label_content(request: LabelRequest):
    """
    Evaluate single content against a policy.
    For bulk processing, use /batch instead.
    """
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        prompt = build_prompt(request.policy, request.content)
        outputs = llm.generate([prompt], SAMPLING_PARAMS)
        raw_output = outputs[0].outputs[0].text.strip()
        label_value = parse_label(raw_output)
        
        return LabelResponse(label=label_value, raw_output=raw_output)
    except Exception as e:
        logger.exception("Error during inference")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=BatchResponse)
async def batch_label(request: BatchRequest):
    """
    Classify multiple items against a single policy using vLLM batching.
    
    vLLM processes all items in parallel using continuous batching,
    providing significant throughput improvements over sequential processing.
    
    No practical limit on batch size - vLLM handles memory management.
    Recommended: 100-500 items per request for optimal throughput.
    """
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.items) == 0:
        return BatchResponse(
            results=[],
            processed=0,
            elapsed_seconds=0.0,
            items_per_second=0.0
        )
    
    start_time = time.time()
    
    try:
        # Build all prompts
        prompts = [build_prompt(request.policy, item.content) for item in request.items]
        
        # vLLM processes all prompts in parallel with continuous batching
        outputs = llm.generate(prompts, SAMPLING_PARAMS)
        
        # Parse results
        results = []
        for item, output in zip(request.items, outputs):
            raw_output = output.outputs[0].text.strip()
            label_value = parse_label(raw_output)
            results.append(BatchResultItem(
                id=item.id,
                label=label_value,
                raw_output=raw_output
            ))
        
        elapsed = time.time() - start_time
        items_per_second = len(results) / elapsed if elapsed > 0 else 0.0
        
        return BatchResponse(
            results=results,
            processed=len(results),
            elapsed_seconds=round(elapsed, 3),
            items_per_second=round(items_per_second, 2)
        )
    
    except Exception as e:
        logger.exception("Error during batch inference")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
