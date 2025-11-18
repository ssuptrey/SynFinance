"""
Streamlit Cloud Entry Point
This file exists at the root level for easier Streamlit Cloud deployment.
It simply imports and runs the main app from src/app.py
"""
import sys
import os

# Add src directory to Python path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now run the main app
exec(open('src/app.py').read())
