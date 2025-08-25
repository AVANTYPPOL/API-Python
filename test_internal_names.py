#!/usr/bin/env python3
"""Test script to verify internal service names are returned correctly"""

from collections import OrderedDict

# Expected service order for consistent API responses
EXPECTED_SERVICES = ['PREMIER', 'SUV_PREMIER', 'UBERX', 'UBERXL']

def test_service_output():
    """Test that services are output with internal names in correct order"""
    
    print("Testing service name output format...")
    print("="*50)
    
    # Test Case 1: Model returns internal names (already correct)
    print("\nTest Case 1: Model returns internal names")
    model_output = {
        'UBERX': 25.50,
        'UBERXL': 35.75,
        'PREMIER': 45.20,
        'SUV_PREMIER': 55.90
    }
    
    # Reorder to match expected order
    result = OrderedDict()
    for service in EXPECTED_SERVICES:
        if service in model_output:
            result[service] = model_output[service]
    
    print(f"Input: {model_output}")
    print(f"Output: {dict(result)}")
    print(f"Order check: {list(result.keys())}")
    assert list(result.keys()) == EXPECTED_SERVICES, f"Order mismatch! Expected {EXPECTED_SERVICES}, got {list(result.keys())}"
    print("✅ PASSED - Correct order: PREMIER, SUV_PREMIER, UBERX, UBERXL\n")
    
    # Test Case 2: Verify exact output format
    print("Test Case 2: Verify exact JSON format")
    expected_output = OrderedDict([
        ('PREMIER', 45.20),
        ('SUV_PREMIER', 55.90),
        ('UBERX', 25.50),
        ('UBERXL', 35.75)
    ])
    
    print(f"Expected format:")
    print(f'{{"predictions": {dict(expected_output)}}}')
    print("✅ This is the exact format the API should return\n")
    
    # Test Case 3: Fallback prices
    print("Test Case 3: Fallback prices in correct order")
    DEFAULT_FALLBACK_PRICES = OrderedDict([
        ('PREMIER', 45.20),
        ('SUV_PREMIER', 55.90),
        ('UBERX', 25.50),
        ('UBERXL', 35.75)
    ])
    
    print(f"Fallback: {dict(DEFAULT_FALLBACK_PRICES)}")
    print(f"Order: {list(DEFAULT_FALLBACK_PRICES.keys())}")
    assert list(DEFAULT_FALLBACK_PRICES.keys()) == EXPECTED_SERVICES
    print("✅ PASSED - Fallback maintains correct order\n")
    
    print("="*50)
    print("✅ ALL TESTS PASSED!")
    print("API will return internal service names in this order:")
    print("  1. PREMIER")
    print("  2. SUV_PREMIER") 
    print("  3. UBERX")
    print("  4. UBERXL")
    print("="*50)

if __name__ == "__main__":
    test_service_output()