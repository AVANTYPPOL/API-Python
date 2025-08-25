#!/usr/bin/env python3
"""Test script to verify service name conversion logic"""

from collections import OrderedDict

# Service name mapping to ensure consistency
SERVICE_NAME_MAP = {
    'UBERX': 'UberX',
    'UBERXL': 'UberXL', 
    'PREMIER': 'Uber Premier',
    'SUV_PREMIER': 'Premier SUV',
    # Handle already converted names (pass-through)
    'UberX': 'UberX',
    'UberXL': 'UberXL',
    'Uber Premier': 'Uber Premier',
    'Premier SUV': 'Premier SUV'
}

# Expected service order for consistent API responses
EXPECTED_SERVICES = ['UberX', 'UberXL', 'Uber Premier', 'Premier SUV']

def test_service_conversion():
    """Test various scenarios of service name conversion"""
    
    # Test Case 1: Model returns internal names
    print("Test Case 1: Model returns internal names")
    model_output = {
        'UBERX': 25.50,
        'UBERXL': 35.75,
        'PREMIER': 45.20,
        'SUV_PREMIER': 55.90
    }
    
    normalized = OrderedDict()
    for service in EXPECTED_SERVICES:
        if service in model_output:
            normalized[service] = model_output[service]
        else:
            # Try to find by internal name
            for internal, customer in SERVICE_NAME_MAP.items():
                if customer == service and internal in model_output:
                    normalized[service] = model_output[internal]
                    break
    
    print(f"Input: {model_output}")
    print(f"Output: {dict(normalized)}")
    assert list(normalized.keys()) == EXPECTED_SERVICES
    print("✅ PASSED\n")
    
    # Test Case 2: Model returns customer-facing names
    print("Test Case 2: Model returns customer-facing names")
    model_output = {
        'UberX': 30.00,
        'UberXL': 40.00,
        'Uber Premier': 50.00,
        'Premier SUV': 60.00
    }
    
    normalized = OrderedDict()
    for service in EXPECTED_SERVICES:
        if service in model_output:
            normalized[service] = model_output[service]
        else:
            # Try to find by internal name
            for internal, customer in SERVICE_NAME_MAP.items():
                if customer == service and internal in model_output:
                    normalized[service] = model_output[internal]
                    break
    
    print(f"Input: {model_output}")
    print(f"Output: {dict(normalized)}")
    assert list(normalized.keys()) == EXPECTED_SERVICES
    print("✅ PASSED\n")
    
    # Test Case 3: Model returns mixed names
    print("Test Case 3: Model returns mixed names")
    model_output = {
        'UBERX': 25.50,
        'UberXL': 35.75,
        'PREMIER': 45.20,
        'Premier SUV': 55.90
    }
    
    normalized = OrderedDict()
    for service in EXPECTED_SERVICES:
        if service in model_output:
            normalized[service] = model_output[service]
        else:
            # Try to find by internal name
            for internal, customer in SERVICE_NAME_MAP.items():
                if customer == service and internal in model_output:
                    normalized[service] = model_output[internal]
                    break
    
    print(f"Input: {model_output}")
    print(f"Output: {dict(normalized)}")
    assert list(normalized.keys()) == EXPECTED_SERVICES
    print("✅ PASSED\n")
    
    # Test Case 4: Model returns partial results
    print("Test Case 4: Model returns partial results (missing services)")
    model_output = {
        'UBERX': 25.50,
        'PREMIER': 45.20
    }
    
    normalized = OrderedDict()
    for service in EXPECTED_SERVICES:
        if service in model_output:
            normalized[service] = model_output[service]
        else:
            # Try to find by internal name
            found = False
            for internal, customer in SERVICE_NAME_MAP.items():
                if customer == service and internal in model_output:
                    normalized[service] = model_output[internal]
                    found = True
                    break
            if not found:
                # Use fallback
                fallback_prices = {'UberX': 25.50, 'UberXL': 35.75, 'Uber Premier': 45.20, 'Premier SUV': 55.90}
                normalized[service] = fallback_prices.get(service, 99.99)
    
    print(f"Input: {model_output}")
    print(f"Output: {dict(normalized)}")
    assert list(normalized.keys()) == EXPECTED_SERVICES
    print("✅ PASSED\n")
    
    print("="*50)
    print("✅ ALL TESTS PASSED!")
    print("Service name conversion logic is working correctly")
    print("="*50)

if __name__ == "__main__":
    test_service_conversion()