# Installation Guide

Complete installation instructions for SynFinance on all platforms.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Database Setup](#database-setup)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Updating](#updating)
- [Uninstallation](#uninstallation)

---

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|------------|
| **Operating System** | Windows 10+, macOS 10.15+, Linux (Ubuntu 20.04+, Debian 10+, RHEL 8+) |
| **Python** | 3.10 or higher |
| **PostgreSQL** | 12 or higher |
| **RAM** | 4GB |
| **Storage** | 5GB free space |
| **CPU** | 2 cores |

### Recommended Specifications

| Component | Recommendation |
|-----------|---------------|
| **Operating System** | Windows 11, macOS 13+, Ubuntu 22.04 LTS |
| **Python** | 3.11 or 3.12 |
| **PostgreSQL** | 15 or 16 |
| **RAM** | 8GB+ |
| **Storage** | 20GB+ SSD |
| **CPU** | 4+ cores |

### Software Prerequisites

1. **Python 3.10+**
   - Download from [python.org](https://www.python.org/downloads/)
   - Verify: `python --version` or `python3 --version`

2. **PostgreSQL 12+**
   - Download from [postgresql.org](https://www.postgresql.org/download/)
   - Verify: `psql --version`

3. **Git** (for source installation)
   - Download from [git-scm.com](https://git-scm.com/downloads)
   - Verify: `git --version`

4. **pip** (usually included with Python)
   - Verify: `pip --version`
   - Upgrade: `python -m pip install --upgrade pip`

---

## Installation Methods

### Method 1: Install from PyPI (Recommended - Coming Soon)

```bash
# Install latest stable version
pip install synfinance

# Install specific version
pip install synfinance==1.0.0

# Install with optional dependencies
pip install synfinance[ml]      # Machine learning features
pip install synfinance[api]     # API server features
pip install synfinance[all]     # All optional features
```

### Method 2: Install from Source (Current Method)

#### Windows (cmd.exe)

```cmd
REM 1. Clone repository
git clone https://github.com/ssuptrey/SynFinance.git
cd SynFinance

REM 2. Create virtual environment
python -m venv .venv

REM 3. Activate virtual environment
.venv\Scripts\activate

REM 4. Upgrade pip
python -m pip install --upgrade pip

REM 5. Install dependencies
pip install -r requirements.txt

REM 6. Verify installation
python -c "import src; print('SynFinance installed successfully!')"
```

#### Windows (PowerShell)

```powershell
# 1. Clone repository
git clone https://github.com/ssuptrey/SynFinance.git
cd SynFinance

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
.venv\Scripts\Activate.ps1

# Note: If you get an execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Verify installation
python -c "import src; print('SynFinance installed successfully!')"
```

#### Linux / macOS

```bash
# 1. Clone repository
git clone https://github.com/ssuptrey/SynFinance.git
cd SynFinance

# 2. Create virtual environment
python3 -m venv .venv

# 3. Activate virtual environment
source .venv/bin/activate

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Verify installation
python -c "import src; print('SynFinance installed successfully!')"
```

### Method 3: Development Installation

For contributors and developers:

```bash
# Clone repository
git clone https://github.com/ssuptrey/SynFinance.git
cd SynFinance

# Create virtual environment
python -m venv .venv

# Activate (Windows cmd)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify
pytest
```

### Method 4: Docker Installation (Coming Soon)

```bash
# Pull Docker image
docker pull synfinance/synfinance:latest

# Run container
docker run -it synfinance/synfinance:latest

# Run with PostgreSQL
docker-compose up
```

---

## Database Setup

### PostgreSQL Installation

#### Windows

1. Download PostgreSQL installer from [postgresql.org](https://www.postgresql.org/download/windows/)
2. Run installer and follow prompts
3. Remember the password you set for the `postgres` user
4. Add PostgreSQL to PATH (usually done automatically)

Verify installation:
```cmd
psql --version
```

#### macOS

Using Homebrew:
```bash
# Install PostgreSQL
brew install postgresql@15

# Start PostgreSQL service
brew services start postgresql@15

# Verify
psql --version
```

#### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Start service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Verify
psql --version
```

### Database Creation

#### Method 1: Using createdb command

```bash
# Create database (as postgres user)
createdb synfinance

# With specific owner
createdb -O your_username synfinance

# Verify
psql -l | grep synfinance
```

#### Method 2: Using psql

```bash
# Connect as postgres user
psql -U postgres

# Create database
CREATE DATABASE synfinance;

# Create user (if needed)
CREATE USER synfinance_user WITH PASSWORD 'your_secure_password';

# Grant privileges
GRANT ALL PRIVILEGES ON DATABASE synfinance TO synfinance_user;

# Exit
\q
```

#### Method 3: Using setup script

```bash
# SynFinance includes a setup script
psql -U postgres -f setup_database.sql
```

### Run Database Migrations

```bash
# Ensure you're in the SynFinance directory with venv activated
cd SynFinance
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Run migrations
python -m alembic upgrade head

# Verify migrations
python -m alembic current
```

---

## Configuration

### Configuration File Setup

SynFinance uses YAML configuration files located in the `config/` directory.

#### 1. Copy Template Configuration

```bash
# Copy default config
cp config/default.yaml config/local.yaml

# Or copy environment-specific template
cp config/development.yaml config/local.yaml
```

#### 2. Edit Configuration

Open `config/local.yaml` and update database settings:

```yaml
database:
  host: localhost
  port: 5432
  name: synfinance
  user: postgres  # or your PostgreSQL username
  password: your_password_here
  pool_size: 10
  max_overflow: 20

logging:
  level: INFO
  file: logs/synfinance.log

generators:
  default_batch_size: 1000
  max_batch_size: 100000
```

#### 3. Environment Variables (Optional)

You can override config settings with environment variables:

**Windows (cmd.exe):**
```cmd
set SYNFINANCE_DB_HOST=localhost
set SYNFINANCE_DB_PORT=5432
set SYNFINANCE_DB_NAME=synfinance
set SYNFINANCE_DB_USER=postgres
set SYNFINANCE_DB_PASSWORD=your_password
```

**Windows (PowerShell):**
```powershell
$env:SYNFINANCE_DB_HOST="localhost"
$env:SYNFINANCE_DB_PORT="5432"
$env:SYNFINANCE_DB_NAME="synfinance"
$env:SYNFINANCE_DB_USER="postgres"
$env:SYNFINANCE_DB_PASSWORD="your_password"
```

**Linux/macOS:**
```bash
export SYNFINANCE_DB_HOST=localhost
export SYNFINANCE_DB_PORT=5432
export SYNFINANCE_DB_NAME=synfinance
export SYNFINANCE_DB_USER=postgres
export SYNFINANCE_DB_PASSWORD=your_password
```

#### 4. Using .env File

Create a `.env` file in the repository root:

```bash
# Database Configuration
SYNFINANCE_DB_HOST=localhost
SYNFINANCE_DB_PORT=5432
SYNFINANCE_DB_NAME=synfinance
SYNFINANCE_DB_USER=postgres
SYNFINANCE_DB_PASSWORD=your_password

# Logging
SYNFINANCE_LOG_LEVEL=INFO

# Performance
SYNFINANCE_BATCH_SIZE=1000
```

Load automatically:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Verification

### Verify Python Installation

```bash
# Check Python version
python --version

# Should output: Python 3.10.x or higher
```

### Verify Virtual Environment

```bash
# On Windows
where python
# Should show: E:\SynFinance\.venv\Scripts\python.exe

# On Linux/macOS
which python
# Should show: /path/to/SynFinance/.venv/bin/python
```

### Verify Dependencies

```bash
# List installed packages
pip list

# Check specific packages
pip show sqlalchemy
pip show pandas
pip show numpy
```

### Verify Database Connection

```python
# Create test file: test_connection.py
from src.database.manager import DatabaseManager
from src.core.config import load_config

config = load_config()
db = DatabaseManager(config)

try:
    db.connect()
    print("✓ Database connection successful!")
    db.disconnect()
except Exception as e:
    print(f"✗ Database connection failed: {e}")
```

Run test:
```bash
python test_connection.py
```

### Verify Generators

```python
# Create test file: test_generators.py
from src.generators.customer_generator import CustomerGenerator
from src.core.config import load_config

config = load_config()
generator = CustomerGenerator(config)

try:
    customers = generator.generate_batch(10)
    print(f"✓ Generated {len(customers)} customers successfully!")
    print(f"  Sample: {customers[0]}")
except Exception as e:
    print(f"✗ Generator failed: {e}")
```

Run test:
```bash
python test_generators.py
```

### Run Test Suite

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test module
pytest tests/generators/test_customer_generator.py

# Run with coverage
pytest --cov=src --cov-report=html
```

Expected output:
```
============================ test session starts =============================
collected XXX items

tests/... ✓✓✓✓✓✓✓✓✓✓ [100%]

============================ XXX passed in X.XXs =============================
```

---

## Troubleshooting

### Common Issues

#### Issue 1: ModuleNotFoundError

**Error:**
```
ModuleNotFoundError: No module named 'src'
```

**Solution:**
```bash
# Ensure you're in the repository root
cd SynFinance

# Ensure virtual environment is activated
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Reinstall dependencies
pip install -r requirements.txt
```

#### Issue 2: PostgreSQL Connection Failed

**Error:**
```
psycopg2.OperationalError: could not connect to server
```

**Solutions:**

1. Check PostgreSQL is running:
   ```bash
   # Windows
   sc query postgresql-x64-15
   
   # Linux
   sudo systemctl status postgresql
   
   # macOS
   brew services list | grep postgresql
   ```

2. Verify connection details:
   ```bash
   psql -h localhost -p 5432 -U postgres -d synfinance
   ```

3. Check `pg_hba.conf` authentication:
   ```bash
   # Location varies by OS
   # Windows: C:\Program Files\PostgreSQL\15\data\pg_hba.conf
   # Linux: /etc/postgresql/15/main/pg_hba.conf
   # macOS: /usr/local/var/postgres/pg_hba.conf
   
   # Add this line:
   host    all             all             127.0.0.1/32            md5
   ```

#### Issue 3: Permission Denied (PowerShell)

**Error:**
```
.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled
```

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Issue 4: Alembic Migration Errors

**Error:**
```
alembic.util.exc.CommandError: Can't locate revision identified by 'XXXXX'
```

**Solution:**
```bash
# Reset migrations
python -m alembic downgrade base
python -m alembic upgrade head

# Or recreate database
dropdb synfinance
createdb synfinance
python -m alembic upgrade head
```

#### Issue 5: Out of Memory

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**
```python
# Reduce batch size in config
generators:
  default_batch_size: 100  # Instead of 1000
  
# Or use batch processing
from src.performance import BatchProcessor

processor = BatchProcessor(config, batch_size=1000)
```

### Getting Help

If you encounter issues not covered here:

1. **Check the FAQ:** `docs/guides/FAQ.md`
2. **Search existing issues:** [GitHub Issues](https://github.com/ssuptrey/SynFinance/issues)
3. **Ask in discussions:** [GitHub Discussions](https://github.com/ssuptrey/SynFinance/discussions)
4. **Report a bug:** File a new issue with details

---

## Updating

### Update from PyPI

```bash
# Update to latest version
pip install --upgrade synfinance

# Update to specific version
pip install --upgrade synfinance==1.1.0
```

### Update from Source

```bash
# Navigate to repository
cd SynFinance

# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Pull latest changes
git pull origin main

# Update dependencies
pip install --upgrade -r requirements.txt

# Run new migrations
python -m alembic upgrade head

# Verify
python -c "import src; print('Update successful!')"
```

---

## Uninstallation

### Remove from PyPI

```bash
pip uninstall synfinance
```

### Remove Source Installation

```bash
# Deactivate virtual environment
deactivate

# Remove repository
rm -rf SynFinance  # Linux/macOS
rmdir /s SynFinance  # Windows cmd

# Optional: Remove database
dropdb synfinance
```

### Remove PostgreSQL (Optional)

**Windows:**
- Use "Add or Remove Programs" to uninstall PostgreSQL

**macOS:**
```bash
brew uninstall postgresql@15
brew cleanup
```

**Linux:**
```bash
sudo apt remove postgresql postgresql-contrib
sudo apt autoremove
```

---

## Next Steps

After successful installation:

1. **[Getting Started Guide](GETTING_STARTED.md)** - Generate your first dataset
2. **[User Guide](USER_GUIDE.md)** - Learn all features
3. **[API Reference](../api/API_REFERENCE.md)** - Explore the API
4. **[Examples](../../examples/)** - Run demo scripts

**Installation complete!** Ready to generate synthetic data.
