"""Fix imports in test files"""
import pathlib

test_files = list(pathlib.Path('tests').rglob('*.py'))
fixed_count = 0

for file_path in test_files:
    content = file_path.read_text(encoding='utf-8')
    modified = False
    
    if 'from customer_generator import' in content:
        content = content.replace('from customer_generator import', 'from src.customer_generator import')
        modified = True
    
    if 'from customer_profile import' in content:
        content = content.replace('from customer_profile import', 'from src.customer_profile import')
        modified = True
    
    if modified:
        file_path.write_text(content, encoding='utf-8')
        print(f'Fixed: {file_path}')
        fixed_count += 1

print(f'\nTotal files fixed: {fixed_count}')
