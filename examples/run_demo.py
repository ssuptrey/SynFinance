"""
SynFinance Interactive Demo Launcher
====================================

Easy way to discover and run all available demos!

Usage:
    python run_demo.py              # Interactive menu
    python run_demo.py --list       # List all demos
    python run_demo.py --week10     # Run Week 10 suite
    python run_demo.py --demo 5     # Run specific demo #5
"""

import os
import sys
import subprocess
from typing import Dict, List, Tuple

# ASCII art banner (no emojis for Windows compatibility)
BANNER = """
================================================================================
   _____            ______ _                            
  / ____|          |  ____(_)                           
 | (___  _   _ _ __| |__   _ _ __   __ _ _ __   ___ ___ 
  \\___ \\| | | | '_ \\  __| | | '_ \\ / _` | '_ \\ / __/ _ \\
  ____) | |_| | | | | |    | | | | | (_| | | | | (_|  __/
 |_____/ \\__, |_| |_|_|    |_|_| |_|\\__,_|_| |_|\\___\\___|
          __/ |                                          
         |___/                DEMO LAUNCHER              
================================================================================
"""

# Demo categories
DEMOS = {
    "week10": {
        "name": "Week 10: Analytics & Performance Suite (RECOMMENDED)",
        "description": "Newest and most comprehensive demos - START HERE!",
        "demos": [
            ("week10_analytics/1_statistical_analysis_demo.py", "Statistical Analysis", "30s", "Correlation, regression, distributions"),
            ("week10_analytics/2_visualization_demo.py", "Data Visualization", "45s", "8 chart types, fraud patterns"),
            ("week10_analytics/3_reporting_demo.py", "Executive Reporting", "20s", "Multi-format reports"),
            ("week10_analytics/4_fraud_detection_demo.py", "Advanced Fraud Detection", "60s", "99% accuracy, ensemble models"),
            ("week10_analytics/5_performance_optimization_demo.py", "Performance Optimization", "40s", "85% memory reduction, 40x speedup"),
        ]
    },
    "fraud": {
        "name": "Fraud Detection Examples",
        "description": "Production-ready fraud detection with ML models",
        "demos": [
            ("fraud_detection/demo_fraud_detection.py", "Complete Fraud Detection", "60s", "Best starting point"),
            ("fraud_detection/demo_all_fraud_patterns.py", "All Fraud Patterns", "45s", "Pattern detection methods"),
            ("fraud_detection/demo_ensemble_models.py", "Ensemble ML Models", "90s", "Random Forest, XGBoost, etc."),
            ("fraud_detection/demo_geographic_patterns.py", "Geographic Patterns", "30s", "Location-based fraud"),
            ("fraud_detection/demo_merchant_ecosystem.py", "Merchant Analysis", "35s", "Merchant network fraud"),
            ("fraud_detection/analyze_fraud_patterns.py", "Deep Pattern Analysis", "50s", "Advanced analysis"),
            ("fraud_detection/analyze_anomaly_patterns.py", "Anomaly Detection", "40s", "Outlier detection"),
            ("fraud_detection/optimize_fraud_models.py", "Model Optimization", "120s", "Hyperparameter tuning"),
            ("fraud_detection/train_fraud_detector.py", "Train Custom Models", "90s", "Model training"),
        ]
    },
    "api": {
        "name": "API & Monitoring Examples",
        "description": "API usage, observability, and multi-tenancy",
        "demos": [
            ("api_examples/api_demo.py", "API Basics", "10s", "Core API patterns"),
            ("api_examples/api_integration_example.py", "API Integration", "30s", "Complete integration"),
            ("api_examples/demo_api_versioning.py", "API Versioning", "15s", "Version compatibility"),
            ("api_examples/demo_tenancy.py", "Multi-Tenancy", "20s", "Tenant isolation"),
            ("api_examples/demo_qa_framework.py", "QA Framework", "25s", "Testing framework"),
            ("api_examples/monitoring_demo.py", "App Monitoring", "25s", "Metrics and logs"),
            ("api_examples/real_time_monitoring.py", "Real-Time Metrics", "30s", "Live monitoring"),
            ("api_examples/demo_observability.py", "Full Observability", "40s", "Logs, metrics, traces"),
        ]
    },
    "data": {
        "name": "Data Generation & ML Pipeline",
        "description": "Generate datasets and build ML pipelines",
        "demos": [
            ("data_generation/complete_ml_pipeline.py", "Complete ML Pipeline", "60s", "End-to-end pipeline"),
            ("data_generation/generate_fraud_training_data.py", "Training Data Generation", "20s", "Fraud training data"),
            ("data_generation/generate_anomaly_dataset.py", "Anomaly Dataset", "15s", "Labeled anomalies"),
            ("data_generation/generate_anomaly_ml_features.py", "ML Features", "25s", "Feature extraction"),
            ("data_generation/generate_combined_features.py", "Combined Features", "30s", "Complete features"),
            ("data_generation/batch_processing_example.py", "Batch Processing", "20s", "Process in batches"),
        ]
    },
    "tutorials": {
        "name": "Learning Tutorials",
        "description": "Step-by-step beginner-friendly tutorials",
        "demos": [
            ("tutorials/fraud_detection_tutorial.py", "Fraud Detection Tutorial", "15min", "Beginner tutorial"),
        ]
    }
}


def print_banner():
    """Print welcome banner."""
    print(BANNER)


def print_menu():
    """Print interactive menu."""
    print("\nWhat would you like to explore?\n")
    
    options = [
        ("1", "Week 10 Analytics Suite", "(Recommended - Newest demos)"),
        ("2", "Fraud Detection Demos", "(9 fraud detection examples)"),
        ("3", "API & Monitoring Demos", "(8 API/observability examples)"),
        ("4", "Data Generation Demos", "(6 data/ML pipeline examples)"),
        ("5", "Tutorials", "(Beginner-friendly learning)"),
        ("", "", ""),
        ("L", "List all demos", "(See complete catalog)"),
        ("H", "Help", "(Usage instructions)"),
        ("Q", "Quit", ""),
    ]
    
    for key, name, desc in options:
        if key:
            print(f"  [{key}] {name:30s} {desc}")
        else:
            print()


def list_all_demos():
    """List all available demos."""
    print("\nComplete Demo Catalog:\n")
    
    demo_num = 1
    for category_key, category in DEMOS.items():
        print(f"\n{category['name']}")
        print(f"  {category['description']}")
        print(f"  {'-' * 70}")
        
        for path, name, runtime, features in category["demos"]:
            print(f"  [{demo_num:2d}] {name:35s} ({runtime:6s}) - {features}")
            demo_num += 1
        
        print()


def get_demo_by_number(num: int) -> Tuple[str, str]:
    """Get demo path and name by number."""
    demo_num = 1
    for category_key, category in DEMOS.items():
        for path, name, runtime, features in category["demos"]:
            if demo_num == num:
                return (path, name)
            demo_num += 1
    
    return (None, None)


def run_demo(script_path: str, demo_name: str):
    """Run a demo script."""
    print(f"\n{'=' * 80}")
    print(f"  Running: {demo_name}")
    print(f"  Script: {script_path}")
    print(f"{'=' * 80}\n")
    
    # Get full path
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(examples_dir, script_path)
    
    if not os.path.exists(full_path):
        print(f"ERROR: Demo script not found: {full_path}")
        print(f"\nNote: Some demos may need to be moved to their category folders.")
        print(f"Check if the file exists in the examples/ directory.")
        return False
    
    # Run the demo
    try:
        # Use same Python interpreter
        python_exe = sys.executable
        
        result = subprocess.run(
            [python_exe, full_path],
            cwd=examples_dir,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n{'=' * 80}")
            print(f"  Demo completed successfully!")
            print(f"{'=' * 80}\n")
            return True
        else:
            print(f"\n{'=' * 80}")
            print(f"  Demo exited with code: {result.returncode}")
            print(f"{'=' * 80}\n")
            return False
            
    except KeyboardInterrupt:
        print(f"\n\nDemo interrupted by user.")
        return False
    except Exception as e:
        print(f"\nERROR running demo: {e}")
        return False


def run_category(category_key: str):
    """Run all demos in a category."""
    if category_key not in DEMOS:
        print(f"ERROR: Unknown category: {category_key}")
        return
    
    category = DEMOS[category_key]
    print(f"\n{'=' * 80}")
    print(f"  {category['name']}")
    print(f"  {category['description']}")
    print(f"{'=' * 80}\n")
    
    print(f"Running {len(category['demos'])} demos in sequence...\n")
    
    for i, (path, name, runtime, features) in enumerate(category["demos"], 1):
        print(f"\n[{i}/{len(category['demos'])}] {name}")
        print(f"Expected runtime: {runtime}")
        print(f"Features: {features}\n")
        
        input("Press Enter to start this demo (or Ctrl+C to skip)...")
        
        if not run_demo(path, name):
            choice = input(f"\nDemo failed. Continue with next demo? (y/n): ")
            if choice.lower() != 'y':
                break
    
    print(f"\n{'=' * 80}")
    print(f"  Category completed: {category['name']}")
    print(f"{'=' * 80}\n")


def show_help():
    """Show help information."""
    print("""
Usage:
------

Interactive Mode:
    python run_demo.py
    
    Shows menu to select demos by category or number.

Command-Line Mode:
    python run_demo.py --list              # List all demos
    python run_demo.py --week10            # Run Week 10 suite
    python run_demo.py --fraud             # Run fraud detection demos
    python run_demo.py --api               # Run API demos
    python run_demo.py --data              # Run data generation demos
    python run_demo.py --demo 5            # Run demo #5
    python run_demo.py --help              # Show this help

Examples:
---------

# Interactive menu
python run_demo.py

# Run best demos (Week 10)
python run_demo.py --week10

# List all available demos
python run_demo.py --list

# Run specific demo by number
python run_demo.py --demo 1

Categories:
-----------

week10    - Week 10 Analytics Suite (5 demos) - RECOMMENDED
fraud     - Fraud Detection (9 demos)
api       - API & Monitoring (8 demos)
data      - Data Generation & ML (6 demos)
tutorials - Learning Tutorials (1 demo)

Tips:
-----

1. Start with Week 10 demos (newest and best)
2. Use --list to see all available demos
3. Check examples/output/ for generated files
4. Read demo source code for examples

For more information, see examples/README.md
""")


def interactive_mode():
    """Run in interactive menu mode."""
    print_banner()
    print("Welcome to the SynFinance Demo Launcher!")
    print("\nExplore 30+ demos organized by category.")
    
    while True:
        print_menu()
        
        choice = input("\nEnter your choice: ").strip().upper()
        
        if choice == 'Q':
            print("\nThank you for exploring SynFinance! Goodbye.\n")
            break
        
        elif choice == 'L':
            list_all_demos()
        
        elif choice == 'H':
            show_help()
        
        elif choice == '1':
            run_category("week10")
        
        elif choice == '2':
            run_category("fraud")
        
        elif choice == '3':
            run_category("api")
        
        elif choice == '4':
            run_category("data")
        
        elif choice == '5':
            run_category("tutorials")
        
        elif choice.isdigit():
            demo_num = int(choice)
            path, name = get_demo_by_number(demo_num)
            if path:
                run_demo(path, name)
            else:
                print(f"\nInvalid demo number: {demo_num}")
                print("Use 'L' to list all demos.")
        
        else:
            print(f"\nInvalid choice: {choice}")
            print("Please enter 1-5, L, H, or Q.")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SynFinance Interactive Demo Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_demo.py              # Interactive menu
  python run_demo.py --list       # List all demos
  python run_demo.py --week10     # Run Week 10 suite
  python run_demo.py --demo 5     # Run demo #5
        """
    )
    
    parser.add_argument('--list', action='store_true', help='List all available demos')
    parser.add_argument('--week10', action='store_true', help='Run Week 10 analytics suite')
    parser.add_argument('--fraud', action='store_true', help='Run fraud detection demos')
    parser.add_argument('--api', action='store_true', help='Run API/monitoring demos')
    parser.add_argument('--data', action='store_true', help='Run data generation demos')
    parser.add_argument('--tutorials', action='store_true', help='Run tutorials')
    parser.add_argument('--demo', type=int, metavar='N', help='Run specific demo by number')
    parser.add_argument('--help-usage', action='store_true', help='Show detailed help')
    
    args = parser.parse_args()
    
    # Handle command-line arguments
    if args.list:
        print_banner()
        list_all_demos()
    
    elif args.help_usage:
        print_banner()
        show_help()
    
    elif args.week10:
        print_banner()
        run_category("week10")
    
    elif args.fraud:
        print_banner()
        run_category("fraud")
    
    elif args.api:
        print_banner()
        run_category("api")
    
    elif args.data:
        print_banner()
        run_category("data")
    
    elif args.tutorials:
        print_banner()
        run_category("tutorials")
    
    elif args.demo:
        print_banner()
        path, name = get_demo_by_number(args.demo)
        if path:
            run_demo(path, name)
        else:
            print(f"ERROR: Invalid demo number: {args.demo}")
            print("\nUse --list to see all available demos.")
            sys.exit(1)
    
    else:
        # No arguments - run interactive mode
        interactive_mode()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
