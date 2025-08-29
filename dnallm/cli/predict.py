#!/usr/bin/env python3
"""
Prediction CLI module for DNALLM package.
"""

import click
import sys
from pathlib import Path

def main():
    """Main prediction function"""
    from ..inference import DNAPredictor
    from ..configuration import load_config
    
    # This function will be called from the main CLI
    # For standalone usage, we can add command line parsing here
    if len(sys.argv) > 1:
        # Simple command line interface for standalone usage
        if len(sys.argv) < 3:
            print("Usage: python -m dnallm.cli.predict <config_file> <model_path>")
            sys.exit(1)
        
        config_file = sys.argv[1]
        model_path = sys.argv[2]
        
        try:
            config_dict = load_config(config_file)
            predictor = DNAPredictor(config_dict)
            results = predictor.predict()
            print(results)
        except Exception as e:
            print(f"Prediction failed: {e}")
            sys.exit(1)
    else:
        print("DNALLM Prediction Module")
        print("Use the main CLI: dnallm predict --help")

if __name__ == '__main__':
    main()
