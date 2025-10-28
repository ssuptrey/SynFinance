"""
SynFinance - Synthetic Financial Transaction Generator for Indian Market
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="synfinance",
    version="0.5.0",  # Week 4 Days 3-4 complete - Advanced fraud patterns
    author="SynFinance Development Team",
    author_email="dev@synfinance.example.com",
    description="Synthetic financial transaction data generator for Indian market with fraud detection features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/synfinance",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/synfinance/issues",
        "Documentation": "https://github.com/yourusername/synfinance/blob/main/docs/INDEX.md",
        "Source Code": "https://github.com/yourusername/synfinance",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Testing",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
        ],
        "ml": [
            "scikit-learn>=1.3.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "api": [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "pydantic>=2.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "synfinance=src.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["utils/*.py", "generators/*.py", "models/*.py"],
    },
    keywords=[
        "synthetic data",
        "financial transactions",
        "fraud detection",
        "machine learning",
        "data generation",
        "indian market",
        "banking",
        "fintech",
    ],
    zip_safe=False,
)
