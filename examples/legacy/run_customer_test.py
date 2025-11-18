"""
Simple test runner for customer validation
"""
import sys
sys.path.insert(0, 'src')

from tests.test_customer_generation import validate_customer_generation

if __name__ == "__main__":
    success, stats = validate_customer_generation()
