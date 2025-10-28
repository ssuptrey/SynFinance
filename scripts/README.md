# SynFinance Scripts

Utility scripts for maintenance, refactoring, and automation tasks.

## Available Scripts

### 1. `refactor_script.py`
**Purpose:** Automated code refactoring and cleanup utility.

**Features:**
- Remove emoji characters from code files
- Clean up whitespace and formatting
- Standardize import statements
- Update file headers and docstrings

**Usage:**
```bash
python scripts/refactor_script.py
```

**Options:**
- `--dry-run`: Preview changes without modifying files
- `--path <path>`: Target specific directory or file
- `--backup`: Create backup files before refactoring

**Example:**
```bash
# Preview changes
python scripts/refactor_script.py --dry-run

# Refactor specific directory
python scripts/refactor_script.py --path src/generators/

# Refactor with backup
python scripts/refactor_script.py --backup
```

---

### 2. `run.sh` (Unix/Linux/Mac)
**Purpose:** Quick project execution script for Unix-based systems.

**Features:**
- Activate virtual environment
- Set environment variables
- Run application or tests
- Clean up temporary files

**Usage:**
```bash
chmod +x scripts/run.sh
./scripts/run.sh
```

**Commands:**
```bash
./scripts/run.sh               # Run main application
./scripts/run.sh test          # Run test suite
./scripts/run.sh clean         # Clean temporary files
./scripts/run.sh install       # Install dependencies
```

---

### 3. `run.bat` (Windows)
**Purpose:** Quick project execution script for Windows.

**Features:**
- Activate virtual environment
- Set environment variables (PYTHONPATH)
- Run application or tests
- Clean up temporary files

**Usage:**
```cmd
scripts\run.bat
```

**Commands:**
```cmd
scripts\run.bat               # Run main application
scripts\run.bat test          # Run test suite
scripts\run.bat clean         # Clean temporary files
scripts\run.bat install       # Install dependencies
```

**Example:**
```cmd
# Run main application
scripts\run.bat

# Run tests
scripts\run.bat test

# Clean and reinstall
scripts\run.bat clean
scripts\run.bat install
```

---

## Script Details

### refactor_script.py

**Location:** `scripts/refactor_script.py`

**Purpose:** Automated code refactoring tool for maintaining code quality.

**Operations:**
1. **Emoji Removal:** Remove emoji characters from Python files
2. **Whitespace Cleanup:** Standardize indentation and spacing
3. **Import Sorting:** Sort and organize import statements
4. **Docstring Updates:** Add or update file/function docstrings

**Safety Features:**
- Dry-run mode for previewing changes
- Automatic backup creation
- Skip binary files
- Preserve file permissions

**Configuration:**
```python
# Customize refactoring rules in refactor_script.py
RULES = {
    'remove_emojis': True,
    'clean_whitespace': True,
    'sort_imports': True,
    'update_docstrings': False  # Requires manual review
}
```

**Output:**
```
Refactoring: src/generators/temporal_generator.py
  ✓ Removed 5 emoji characters
  ✓ Cleaned 12 whitespace issues
  ✓ Sorted 8 import statements

Summary:
  Files processed: 23
  Changes made: 47
  Backup created: backup_20241021_143025/
```

---

### run.sh

**Location:** `scripts/run.sh`

**Supported Commands:**
```bash
./scripts/run.sh               # Run main application (app.py)
./scripts/run.sh test          # Run pytest test suite
./scripts/run.sh clean         # Remove __pycache__, *.pyc, etc.
./scripts/run.sh install       # Install requirements.txt
./scripts/run.sh format        # Format code with black/autopep8
./scripts/run.sh lint          # Run pylint/flake8
```

**Environment Setup:**
```bash
# Set PYTHONPATH
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Activate virtual environment
source .venv/bin/activate
```

**Customization:**
Edit `scripts/run.sh` to add custom commands:
```bash
case "$1" in
    custom)
        echo "Running custom command..."
        python custom_script.py
        ;;
esac
```

---

### run.bat

**Location:** `scripts/run.bat`

**Supported Commands:**
```cmd
scripts\run.bat               # Run main application (app.py)
scripts\run.bat test          # Run pytest test suite
scripts\run.bat clean         # Remove __pycache__, *.pyc, etc.
scripts\run.bat install       # Install requirements.txt
scripts\run.bat format        # Format code with black/autopep8
scripts\run.bat lint          # Run pylint/flake8
```

**Environment Setup:**
```cmd
@echo off
set PYTHONPATH=%CD%\src;%PYTHONPATH%
call .venv\Scripts\activate.bat
```

**Customization:**
Edit `scripts/run.bat` to add custom commands:
```cmd
if "%1"=="custom" (
    echo Running custom command...
    python custom_script.py
    goto :end
)
```

---

## Creating Custom Scripts

### Python Script Template

```python
"""
Script: custom_task.py
Purpose: Brief description of what this script does
"""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def main():
    """Main script logic"""
    print("Running custom task...")
    
    # Your code here
    
    print("Task completed!")

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
python scripts/custom_task.py
```

---

### Shell Script Template (Unix)

```bash
#!/bin/bash
# Script: custom_task.sh
# Purpose: Brief description

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Run task
echo "Running custom task..."
python "$PROJECT_ROOT/scripts/custom_task.py"
echo "Task completed!"
```

**Usage:**
```bash
chmod +x scripts/custom_task.sh
./scripts/custom_task.sh
```

---

### Batch Script Template (Windows)

```cmd
@echo off
REM Script: custom_task.bat
REM Purpose: Brief description

REM Get script directory
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

REM Activate virtual environment
call "%PROJECT_ROOT%\.venv\Scripts\activate.bat"

REM Set PYTHONPATH
set PYTHONPATH=%PROJECT_ROOT%\src;%PYTHONPATH%

REM Run task
echo Running custom task...
python "%PROJECT_ROOT%\scripts\custom_task.py"
echo Task completed!

pause
```

**Usage:**
```cmd
scripts\custom_task.bat
```

---

## Common Tasks

### Task 1: Clean Project
```bash
# Unix/Linux/Mac
./scripts/run.sh clean

# Windows
scripts\run.bat clean
```

**What it cleans:**
- `__pycache__` directories
- `*.pyc` files
- `*.pyo` files
- `.pytest_cache`
- `.coverage` files
- Build artifacts

---

### Task 2: Run Tests
```bash
# Unix/Linux/Mac
./scripts/run.sh test

# Windows
scripts\run.bat test
```

**Options:**
```bash
# Run specific test suite
./scripts/run.sh test tests/generators/

# Run with coverage
./scripts/run.sh test --cov=src

# Run with verbose output
./scripts/run.sh test -v
```

---

### Task 3: Install Dependencies
```bash
# Unix/Linux/Mac
./scripts/run.sh install

# Windows
scripts\run.bat install
```

---

### Task 4: Format Code
```bash
# Unix/Linux/Mac
./scripts/run.sh format

# Windows
scripts\run.bat format
```

**Formatters Used:**
- `black` - Python code formatter
- `autopep8` - PEP 8 compliance
- `isort` - Import sorting

---

### Task 5: Lint Code
```bash
# Unix/Linux/Mac
./scripts/run.sh lint

# Windows
scripts\run.bat lint
```

**Linters Used:**
- `pylint` - Python code analysis
- `flake8` - Style guide enforcement
- `mypy` - Type checking

---

## Best Practices

### 1. **Version Control**
- Always create backups before running destructive scripts
- Use `--dry-run` mode to preview changes
- Commit changes before running refactoring scripts

### 2. **Documentation**
- Add clear comments to custom scripts
- Update README when adding new scripts
- Document all command-line arguments

### 3. **Error Handling**
```python
import sys

try:
    # Script logic
    main()
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
```

### 4. **Logging**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info("Script started")
```

---

## Troubleshooting

### Script Not Found
```bash
# Ensure you're in project root
cd e:\SynFinance

# Check script permissions (Unix)
ls -l scripts/
chmod +x scripts/run.sh
```

### Python Path Issues
```bash
# Set PYTHONPATH manually
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"  # Unix
set PYTHONPATH=%CD%\src;%PYTHONPATH%         # Windows
```

### Virtual Environment Not Activated
```bash
# Activate manually
source .venv/bin/activate     # Unix
.venv\Scripts\activate.bat    # Windows
```

### Script Execution Failed
```bash
# Check Python version
python --version  # Should be 3.13.3

# Check dependencies
pip list

# Reinstall if needed
pip install -r requirements.txt
```

---

## Maintenance Schedule

### Daily
- Run tests before committing (`run.sh test`)
- Clean temporary files (`run.sh clean`)

### Weekly
- Run linting (`run.sh lint`)
- Format code (`run.sh format`)
- Review refactoring suggestions

### Monthly
- Update dependencies
- Review and update scripts
- Clean up unused scripts

---

## Documentation

- **Quick Start:** See [docs/guides/QUICKSTART.md](../docs/guides/QUICKSTART.md)
- **Architecture:** See [docs/technical/ARCHITECTURE.md](../docs/technical/ARCHITECTURE.md)
- **Contributing:** See [CONTRIBUTING.md](../CONTRIBUTING.md)

---

## Contributing

Found a bug or want to add a new script? See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
