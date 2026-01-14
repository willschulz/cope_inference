#!/usr/bin/env python3
"""
CoPE-A-9B Inference Service

A FastAPI service for content policy evaluation using the CoPE-A-9B model.
Runs quantized inference on NVIDIA L4 GPU.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Model configuration
BASE_MODEL_ID = "google/gemma-2-9b"
ADAPTER_MODEL_ID = "zentropi-ai/cope-a-9b"

# Global model state
model = None
tokenizer = None


def build_prompt(policy: str, content: str) -> str:
    """
    Build the CoPE prompt from policy and content.
    
    Format follows the CoPE-A specification:
    INSTRUCTIONS + POLICY + CONTENT + ANSWER sections.
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
    Load the base model with 4-bit quantization and apply the CoPE adapter.
    """
    global model, tokenizer
    
    logger.info("Configuring 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    logger.info(f"Loading base model: {BASE_MODEL_ID}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    logger.info(f"Loading tokenizer: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
    )
    
    logger.info(f"Applying CoPE adapter: {ADAPTER_MODEL_ID}")
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    
    model.eval()
    
    # Log memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    logger.info("Model loaded successfully!")
    return model, tokenizer


def label(policy: str, content: str) -> dict:
    """
    Classify content against a policy.
    
    Args:
        policy: The content policy definition (CoPE format)
        content: The content to evaluate
        
    Returns:
        dict with 'label' (0 or 1) and 'raw_output' (the generated token)
    """
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    # Build the prompt
    prompt = build_prompt(policy, content)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate exactly one token
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Extract the generated token (last token)
    generated_token_id = outputs[0, -1].item()
    raw_output = tokenizer.decode([generated_token_id], skip_special_tokens=True).strip()
    
    # Parse to binary label
    if raw_output == "1":
        label_value = 1
    elif raw_output == "0":
        label_value = 0
    else:
        # Fallback: check if the output contains 0 or 1
        if "1" in raw_output:
            label_value = 1
        elif "0" in raw_output:
            label_value = 0
        else:
            logger.warning(f"Unexpected model output: '{raw_output}', defaulting to 0")
            label_value = 0
    
    return {"label": label_value, "raw_output": raw_output}


# Pydantic models for API
class LabelRequest(BaseModel):
    policy: str
    content: str


class LabelResponse(BaseModel):
    label: int
    raw_output: str


class BatchItem(BaseModel):
    id: str = Field(..., description="Unique identifier for this item (e.g., tweet_id)")
    content: str = Field(..., description="Text content to classify")


class BatchRequest(BaseModel):
    policy: str = Field(..., description="The content policy to apply to all items")
    items: List[BatchItem] = Field(..., description="List of items to classify (max 100 per request)")


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


# FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    logger.info("Starting CoPE-A-9B inference service...")
    load_model()
    yield
    logger.info("Shutting down CoPE-A-9B inference service...")


app = FastAPI(
    title="CoPE-A-9B Inference Service",
    description="Content policy evaluation using the CoPE-A-9B model",
    version="1.0.0",
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
        model_loaded=model is not None,
        gpu_available=torch.cuda.is_available(),
        gpu_name=gpu_name,
        gpu_memory_allocated_gb=gpu_memory,
    )


@app.post("/label", response_model=LabelResponse)
async def label_content(request: LabelRequest):
    """
    Evaluate content against a policy.
    
    Returns:
        - label: 0 if content does NOT match any policy labels, 1 if it does
        - raw_output: The raw token generated by the model
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = label(request.policy, request.content)
        return LabelResponse(**result)
    except Exception as e:
        logger.exception("Error during inference")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=BatchResponse)
async def batch_label(request: BatchRequest):
    """
    Classify multiple items against a single policy.
    
    Optimized for bulk processing - send up to 100 items per request.
    The same policy is applied to all items in the batch.
    
    Returns results in the same order as input, with original IDs preserved.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.items) > 100:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch size {len(request.items)} exceeds maximum of 100 items per request"
        )
    
    if len(request.items) == 0:
        return BatchResponse(
            results=[],
            processed=0,
            elapsed_seconds=0.0,
            items_per_second=0.0
        )
    
    start_time = time.time()
    results = []
    
    try:
        for item in request.items:
            result = label(request.policy, item.content)
            results.append(BatchResultItem(
                id=item.id,
                label=result["label"],
                raw_output=result["raw_output"]
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
