#!/usr/bin/env python3
"""
Inference CLI module for DNALLM package.
"""

import sys


def main():
    """Main inference function"""
    from ..inference import DNAInference
    from ..configuration import load_config

    # This function will be called from the main CLI
    # For standalone usage, we can add command line parsing here
    if len(sys.argv) > 1:
        # Simple command line interface for standalone usage
        if len(sys.argv) < 3:
            print(
                                "Usage: python -m dnallm.cli.inference"
                                ""<config_file> <model_path>"
            )
            sys.exit(1)

        config_file = sys.argv[1]
        # model_path = sys.argv[2]  # Not used in current implementation

        try:
            config_dict = load_config(config_file)
            inference_engine = DNAInference(config_dict)
            results = inference_engine.infer()
            print(results)
        except Exception as e:
            print(f"Inference failed: {e}")
            sys.exit(1)
    else:
        print("DNALLM Inference Module")
        print("Use the main CLI: dnallm inference --help")


if __name__ == "__main__":
    main()
