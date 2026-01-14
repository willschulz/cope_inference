#!/usr/bin/env python3
"""Benchmark vLLM throughput with larger batches."""

import requests
import time

API_URL = "http://localhost:8000"

POLICY = """# Criteria

## Definition of Labels

### (HS): Hate Speech

#### Includes
- Slurs targeting protected groups
- Dehumanization
- Calls for violence

#### Excludes
- Counter-speech
- Educational content"""

# Generate 100 test items
TEST_ITEMS = [
    {"id": str(i), "content": f"This is test message number {i}. Hello world, testing the classification system."}
    for i in range(100)
]

def benchmark():
    print(f"Benchmarking with {len(TEST_ITEMS)} items...")
    
    start = time.time()
    response = requests.post(
        f"{API_URL}/batch",
        json={"policy": POLICY, "items": TEST_ITEMS},
        timeout=300
    )
    response.raise_for_status()
    result = response.json()
    
    print(f"Processed: {result['processed']} items")
    print(f"Elapsed: {result['elapsed_seconds']:.2f} seconds")
    print(f"Throughput: {result['items_per_second']:.2f} items/second")
    
    # Verify all labels are 0 (none should be hate speech)
    flagged = sum(1 for r in result['results'] if r['label'] == 1)
    print(f"Flagged as hate speech: {flagged}")

if __name__ == "__main__":
    benchmark()
