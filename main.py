"""
Main Script for Semantic Similarity Project

This script provides a unified interface to run:
- Data exploration
- BERT model training
- LSTM model training

Usage:
    python main.py [options]

Options:
    --explore          Run data exploration
    --bert             Train BERT model
    --lstm             Train LSTM model
    --all              Run all components
    --help             Show this help message
"""

import argparse
import sys
import os
import time
from datetime import datetime


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def run_data_exploration():
    """Run data exploration script."""
    print_header("DATA EXPLORATION")
    try:
        from data_exploration import main as explore_main
        explore_main()
        return True
    except ImportError as e:
        print(f"Error importing data_exploration: {e}")
        return False
    except Exception as e:
        print(f"Error running data exploration: {e}")
        return False


def run_bert_training():
    """Run BERT training script."""
    print_header("BERT MODEL TRAINING")
    try:
        from bert_training import main as bert_main
        bert_main()
        return True
    except ImportError as e:
        print(f"Error importing bert_training: {e}")
        print("Make sure all required packages are installed.")
        return False
    except Exception as e:
        print(f"Error running BERT training: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_lstm_training():
    """Run LSTM training script."""
    print_header("LSTM MODEL TRAINING")
    try:
        from lstm_training import main as lstm_main
        lstm_main()
        return True
    except ImportError as e:
        print(f"Error importing lstm_training: {e}")
        print("Make sure all required packages are installed.")
        return False
    except Exception as e:
        print(f"Error running LSTM training: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn',
        'torch': 'torch',
        'nltk': 'nltk',
    }
    
    missing = []
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print("Warning: The following packages are missing:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nYou can install them with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True


def check_dataset():
    """Check if the dataset is available."""
    dataset_folder = 'stanford-natural-language-inference-corpus'
    required_files = [
        'snli_1.0_train.csv',
        'snli_1.0_dev.csv',
        'snli_1.0_test.csv'
    ]
    
    if not os.path.exists(dataset_folder):
        print(f"Warning: Dataset folder '{dataset_folder}' not found.")
        print("You may need to download the dataset first.")
        print("You can run data exploration to download it, or download manually from Kaggle.")
        return False
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(dataset_folder, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"Warning: The following dataset files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    return True


def main():
    """Main function to orchestrate all components."""
    parser = argparse.ArgumentParser(
        description='Semantic Similarity Project - Main Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --explore              # Run only data exploration
  python main.py --bert                 # Train only BERT model
  python main.py --lstm                 # Train only LSTM model
  python main.py --all                  # Run everything
  python main.py --explore --bert       # Run exploration and BERT training
        """
    )
    
    parser.add_argument(
        '--explore',
        action='store_true',
        help='Run data exploration'
    )
    parser.add_argument(
        '--bert',
        action='store_true',
        help='Train BERT model'
    )
    parser.add_argument(
        '--lstm',
        action='store_true',
        help='Train LSTM model'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all components (exploration, BERT, LSTM)'
    )
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip dependency and dataset checks'
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any([args.explore, args.bert, args.lstm, args.all]):
        parser.print_help()
        return
    
    # Print welcome message
    print("\n" + "="*80)
    print("  SEMANTIC SIMILARITY PROJECT - MAIN SCRIPT")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run checks
    if not args.skip_checks:
        print("\nChecking dependencies...")
        if not check_dependencies():
            print("\nSome dependencies are missing. Continuing anyway...")
        
        print("\nChecking dataset...")
        dataset_available = check_dataset()
        if not dataset_available and not args.explore:
            print("\nDataset not found. You may need to run data exploration first.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting.")
                return
    
    # Determine what to run
    run_explore = args.explore or args.all
    run_bert = args.bert or args.all
    run_lstm = args.lstm or args.all
    
    results = {}
    start_time = time.time()
    
    # Run data exploration
    if run_explore:
        results['exploration'] = run_data_exploration()
        if not results['exploration']:
            print("\nWarning: Data exploration encountered errors.")
    
    # Run BERT training
    if run_bert:
        results['bert'] = run_bert_training()
        if not results['bert']:
            print("\nWarning: BERT training encountered errors.")
    
    # Run LSTM training
    if run_lstm:
        results['lstm'] = run_lstm_training()
        if not results['lstm']:
            print("\nWarning: LSTM training encountered errors.")
    
    # Print summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print("\n" + "="*80)
    print("  EXECUTION SUMMARY")
    print("="*80)
    print(f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
    print("\nResults:")
    for component, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {component.capitalize()}: {status}")
    print("="*80 + "\n")
    
    # Check if all succeeded
    if all(results.values()):
        print("All components completed successfully!")
    else:
        print("Some components encountered errors. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

