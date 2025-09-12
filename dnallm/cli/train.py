#!/usr/bin/env python3
"""
Training CLI module for DNALLM package.
"""

import sys


def main():
    """Main training function"""
    from ..finetune import DNATrainer
    from ..configuration import load_config

    # This function will be called from the main CLI
    # For standalone usage, we can add command line parsing here
    if len(sys.argv) > 1:
        # Simple command line interface for standalone usage
        if len(sys.argv) < 4:
            print(
                                "Usage: python -m dnallm.cli.train"
                                ""<config_file> <model_path> <data_path>"
            )
            sys.exit(1)

        config_file = sys.argv[1]
        # model_path = sys.argv[2]  # Not used in current implementation
        # data_path = sys.argv[3]   # Not used in current implementation

        try:
            config_dict = load_config(config_file)
            trainer = DNATrainer(config_dict)
            trainer.train()
        except Exception as e:
            print(f"Training failed: {e}")
            sys.exit(1)
    else:
        print("DNALLM Training Module")
        print("Use the main CLI: dnallm train --help")


if __name__ == "__main__":
    main()
