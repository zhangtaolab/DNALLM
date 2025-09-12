#!/usr/bin/env python3
"""
DNALLM CLI Launcher

This script provides a convenient way to run DNALLM commands from the
project root.
"""

import sys
from pathlib import Path


def main():
    """Main launcher function"""
    # Add the current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))

    try:
        # Import and run the CLI
        from cli.cli import main as cli_main

        cli_main()
    except ImportError as e:
        print(f"Error importing CLI: {e}")
        print("Please ensure you're running from the project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"CLI execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
