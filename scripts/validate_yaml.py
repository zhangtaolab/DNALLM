#!/usr/bin/env python3
"""Validate all YAML files under example/ can be loaded via load_config()."""

import sys
from pathlib import Path
from dnallm.configuration.configs import load_config

EXAMPLE_DIR = Path(__file__).parent.parent / "example"

yaml_files = sorted(EXAMPLE_DIR.rglob("*.yaml")) + sorted(EXAMPLE_DIR.rglob("*.yml"))
all_ok = True

for f in yaml_files:
    try:
        load_config(str(f))
        print(f"OK: {f.relative_to(EXAMPLE_DIR.parent)}")
    except Exception as e:
        print(f"FAIL: {f.relative_to(EXAMPLE_DIR.parent)} - {e}")
        all_ok = False

print(f"\nTotal: {len(yaml_files)} files")
if all_ok:
    print("All YAML files passed validation.")
    sys.exit(0)
else:
    print("Some YAML files failed validation.")
    sys.exit(1)
