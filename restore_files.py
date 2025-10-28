"""Restore corrupted files from .corrupt.bak backups"""
import re
from pathlib import Path

# List of files to restore
corrupt_files = [
    r"e:\SynFinance\docs\progress\WEEK2_DAY3-4_COMPLETE.md.corrupt.bak",
    r"e:\SynFinance\docs\progress\WEEK3_DAY1_COMPLETE.md.corrupt.bak",
    r"e:\SynFinance\docs\progress\WEEK3_DAY1_PROGRESS.md.corrupt.bak",
    r"e:\SynFinance\docs\technical\ARCHITECTURE.md.corrupt.bak",
    r"e:\SynFinance\docs\technical\WEEK1_COMPLETION_SUMMARY.md.corrupt.bak",
    r"e:\SynFinance\docs\technical\WEEK2_DAY1-2_SUMMARY.md.corrupt.bak",
    r"e:\SynFinance\docs\technical\WEEK2_DAY3-4_SUMMARY.md.corrupt.bak",
    r"e:\SynFinance\docs\technical\WEEK2_DAY5-7_SUMMARY.md.corrupt.bak",
    r"e:\SynFinance\docs\archive\PROJECT_STRUCTURE.md.corrupt.bak",
]

for corrupt_path in corrupt_files:
    corrupt_file = Path(corrupt_path)
    if not corrupt_file.exists():
        print(f"SKIP: {corrupt_file.name} - file not found")
        continue
    
    # Remove .corrupt.bak extension to get original filename
    original_file = Path(str(corrupt_file).replace('.corrupt.bak', ''))
    
    print(f"Restoring: {original_file.name}")
    
    try:
        # Read corrupted content
        with open(corrupt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove corruption pattern: [X][[X]O[X]K[X]][X]
        clean_content = re.sub(r'\[X\]\[\[X\]O\[X\]K\[X\]\]\[X\]', '', content)
        
        # Write clean content
        with open(original_file, 'w', encoding='utf-8', newline='\n') as f:
            f.write(clean_content)
        
        print(f"  [OK] Restored {original_file.name} ({len(clean_content)} chars)")
    
    except Exception as e:
        print(f"  [X] ERROR: {e}")

print("\nRestoration complete!")
