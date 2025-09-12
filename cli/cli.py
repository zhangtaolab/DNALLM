#!/usr/bin/env python3
"""
Main CLI entry point for DNALLM.
This is the root-level CLI that users will run directly.
"""

import sys
from pathlib import Path


def main():
    """Main CLI entry point"""
    # Add the current directory to Python path to import dnallm package
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    sys.path.insert(0, str(project_root))

    try:
        # Import and run the package CLI
        from dnallm.cli import cli

        cli()
    except ImportError as e:
        print(f"Error importing DNALLM package: {e}")
        print(
            "Please ensure the package is properly"
            "installed or run from the project root."
        )
        sys.exit(1)
    except Exception as e:
        print(f"CLI execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
