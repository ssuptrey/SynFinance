#!/usr/bin/env python
"""Script to create clean refactored data_generator.py"""

# Read backup file
with open('src/data_generator_old.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Take convenience functions and main section (lines 705-889)
convenience_functions = lines[704:]

# Create new file with refactored structure
with open('src/data_generator.py', 'w', encoding='utf-8') as f:
    # Write header
    f.write('"""' + '\\n')
    f.write('Data Generator Module for SynFinance' + '\\n')
    f.write('Generates synthetic Indian financial transaction data with customer behavioral consistency' + '\\n')
    f.write('' + '\\n')
    f.write('Week 1, Day 5-7: Customer-Aware Transaction Generation' + '\\n')
    f.write('Focus: Indian AI companies, developer-friendly, 100% scalable' + '\\n')
    f.write('Refactoring: Modular architecture with generators/, utils/, models/' + '\\n')
    f.write('' + '\\n')
    f.write('Key Features:' + '\\n')
    f.write('- Customer behavioral consistency (spending patterns, preferences, timing)' + '\\n')
    f.write('- Indian market patterns (UPI dominance, festival spending, regional preferences)' + '\\n')
    f.write('- Scalable architecture (generator pattern, streaming support)' + '\\n')
    f.write('- Developer-friendly (modular, well-documented, easy to extend)' + '\\n')
    f.write('' + '\\n')
    f.write('Architecture:' + '\\n')
    f.write('- TransactionGenerator class moved to generators/transaction_core.py' + '\\n')
    f.write('- Indian market data moved to utils/indian_data.py' + '\\n')
    f.write('- This file now provides convenient API functions and backward compatibility' + '\\n')
    f.write('"""' + '\\n')
    f.write('' + '\\n')
    
    # Write imports
    f.write('from datetime import datetime' + '\\n')
    f.write('from typing import Tuple, Optional' + '\\n')
    f.write('import pandas as pd' + '\\n')
    f.write('' + '\\n')
    f.write('from customer_profile import CustomerProfile' + '\\n')
    f.write('from customer_generator import CustomerGenerator' + '\\n')
    f.write('' + '\\n')
    f.write('# Import from modular structure' + '\\n')
    f.write('from generators.transaction_core import TransactionGenerator' + '\\n')
    f.write('from utils.indian_data import (' + '\\n')
    f.write('    INDIAN_FESTIVALS,' + '\\n')
    f.write('    INDIAN_MERCHANTS,' + '\\n')
    f.write('    UPI_HANDLES,' + '\\n')
    f.write('    CHAIN_MERCHANTS' + '\\n')
    f.write(')' + '\\n')
    f.write('' + '\\n')
    f.write('' + '\\n')
    f.write('# ============================================================================' + '\\n')
    f.write('# RE-EXPORT FOR BACKWARD COMPATIBILITY' + '\\n')
    f.write('# ============================================================================' + '\\n')
    f.write('' + '\\n')
    f.write('# Re-export so existing imports still work' + '\\n')
    f.write('__all__ = [' + '\\n')
    f.write("    'TransactionGenerator'," + '\\n')
    f.write("    'generate_realistic_dataset'," + '\\n')
    f.write("    'generate_sample_data'," + '\\n')
    f.write("    'INDIAN_FESTIVALS'," + '\\n')
    f.write("    'INDIAN_MERCHANTS'," + '\\n')
    f.write("    'UPI_HANDLES'," + '\\n')
    f.write("    'CHAIN_MERCHANTS'," + '\\n')
    f.write(']' + '\\n')
    f.write('' + '\\n')
    f.write('' + '\\n')
    # Write convenience functions
    f.writelines(convenience_functions)
    
    # Add refactoring note at end
    f.write('\\n# Refactored: Week 1 Days 5-7 + Modular Architecture\\n')

print('✓ Created clean data_generator.py')
print('  - ~265 lines (reduced from 889)')
print('  - Imports TransactionGenerator from generators.transaction_core')
print('  - Imports Indian data from utils.indian_data')
print('  - Maintains backward compatibility')
