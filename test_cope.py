#!/usr/bin/env python3
"""
Test script for CoPE-A-9B inference service.

This script sends test requests to verify the service is working correctly.
"""

import json
import sys
import time

import requests

SERVICE_URL = "http://localhost:8000"

# Sample hate speech policy (minimal example for testing)
SAMPLE_POLICY = """# Criteria

## Overview

This policy classifies content that contains hate speech targeting protected groups.

## Definition of Terms

- Protected Group: Any group defined by race, ethnicity, religion, gender, sexual orientation, disability, or national origin.
- Hate Speech: Expression that attacks, demeans, or incites violence against individuals or groups based on protected characteristics.

## Interpretation of Language

- Consider the overall context and intent of the message
- Sarcasm or irony does not exempt content from classification
- Direct and indirect references should both be considered

## Definition of Labels

### (HS): Hate Speech

#### Includes

- Slurs: Use of derogatory terms targeting protected groups
- Dehumanization: Comparing protected groups to animals, diseases, or subhuman entities
- Calls for Violence: Explicit or implicit calls for harm against protected groups
- Stereotyping: Harmful generalizations about protected groups presented as facts

#### Excludes

- Counter-speech: Content that quotes hate speech to criticize or refute it
- Educational: Academic or journalistic discussion of hate speech
- Self-reference: Members of a group reclaiming or discussing slurs about their own group"""

# Test cases: (content, expected_label, description)
TEST_CASES = [
    (
        "Hello, how are you today? The weather is nice.",
        0,
        "Neutral greeting - should NOT be flagged"
    ),
    (
        "I think we should have a respectful discussion about immigration policy.",
        0,
        "Policy discussion - should NOT be flagged"
    ),
    (
        "Those people are subhuman vermin and should be eliminated.",
        1,
        "Dehumanization + call for violence - SHOULD be flagged"
    ),
]


def check_health():
    """Check if the service is healthy and model is loaded."""
    try:
        response = requests.get(f"{SERVICE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("model_loaded", False), data
        return False, None
    except requests.exceptions.RequestException:
        return False, None


def wait_for_service(timeout=180):
    """Wait for the service to be ready."""
    print(f"Waiting for service to be ready (timeout: {timeout}s)...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        ready, health_data = check_health()
        if ready:
            print(f"Service ready! GPU: {health_data.get('gpu_name')}")
            print(f"GPU Memory Allocated: {health_data.get('gpu_memory_allocated_gb')} GB")
            return True
        time.sleep(5)
        elapsed = int(time.time() - start_time)
        print(f"  ... waiting ({elapsed}s elapsed)")
    
    return False


def test_label(policy: str, content: str) -> dict:
    """Send a label request to the service."""
    response = requests.post(
        f"{SERVICE_URL}/label",
        json={"policy": policy, "content": content},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def run_tests():
    """Run all test cases."""
    print("\n" + "=" * 60)
    print("Running CoPE-A-9B Test Suite")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for i, (content, expected, description) in enumerate(TEST_CASES, 1):
        print(f"\nTest {i}: {description}")
        print(f"  Content: {content[:60]}...")
        
        try:
            result = test_label(SAMPLE_POLICY, content)
            actual = result["label"]
            raw = result["raw_output"]
            
            if actual == expected:
                status = "✓ PASS"
                passed += 1
            else:
                status = "✗ FAIL"
                failed += 1
            
            print(f"  Expected: {expected}, Got: {actual} (raw: '{raw}') - {status}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


def main():
    print("CoPE-A-9B Inference Test")
    print("-" * 40)
    
    # Check if service is running
    ready, health_data = check_health()
    
    if not ready:
        print("Service not ready. Waiting for model to load...")
        if not wait_for_service():
            print("\nERROR: Service did not become ready within timeout.")
            print("Make sure the service is running: ./start.sh")
            print("Check logs: tail -f cope.log")
            sys.exit(1)
    else:
        print(f"Service is ready! GPU: {health_data.get('gpu_name')}")
    
    # Run tests
    success = run_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
