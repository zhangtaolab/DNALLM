#!/usr/bin/env python3
"""
Model Configuration Generator CLI module for DNALLM package.
"""

from pathlib import Path


def main():
    """Main model configuration generator function"""
    try:
        # Import the configuration generator from the root UI module
        # This allows the package to use the same functionality
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from cli.model_config_generator import main as generate_config

        generate_config()

    except ImportError:
        print("DNALLM Model Configuration Generator")
        print("This module requires the root CLI module to be available.")
        print(
            "Please run from the project root: python "
            "cli/model_config_generator.py"
        )
        sys.exit(1)
    except Exception as e:
        print(f"Configuration generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
