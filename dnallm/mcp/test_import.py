#!/usr/bin/env python3
"""
Test script to verify MCP server components can be imported.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from dnallm.mcp.model_config_generator import MCPModelConfigGenerator
    print("✓ Model config generator imported successfully")
    
    from dnallm.mcp.config_manager import ConfigManager
    print("✓ Config manager imported successfully")
    
    from dnallm.mcp.model_manager import ModelManager
    print("✓ Model manager imported successfully")
    
    from dnallm.mcp.utils.validators import validate_dna_sequence
    print("✓ Validators imported successfully")
    
    from dnallm.mcp.utils.formatters import format_prediction_result
    print("✓ Formatters imported successfully")
    
    print("\n🎉 All MCP server components imported successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
