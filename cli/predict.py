#!/usr/bin/env python3
"""
Prediction CLI entry point for DNALLM.
This is the root-level prediction CLI that users will run directly.
"""

import sys
from pathlib import Path

def main():
    """Main prediction CLI entry point"""
    # Add the current directory to Python path to import dnallm package
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    sys.path.insert(0, str(project_root))
    
    try:
        # Import and run the package prediction CLI
        from dnallm.cli import predict_main
        predict_main()
    except ImportError as e:
        print(f"Error importing DNALLM package: {e}")
        print("Please ensure the package is properly installed or run from the project root.")
        sys.exit(1)
    except Exception as e:
        print(f"Prediction CLI execution failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
