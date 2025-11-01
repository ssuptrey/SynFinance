#!/usr/bin/env bash
set -euo pipefail

echo "Installing test dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest yamllint pyyaml

echo "Running pytest..."
pytest tests/ -v --tb=short

echo "Running manifest tests only"
pytest tests/deployment/test_kubernetes.py::TestKubernetesManifests -q

echo "All tests completed."