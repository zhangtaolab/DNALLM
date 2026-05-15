"""Test that all example YAML configs load successfully via load_config()."""

from pathlib import Path

import pytest

from dnallm.configuration.configs import load_config

EXAMPLE_DIR = Path(__file__).parent.parent.parent / "example"


def _get_yaml_files():
    files = list(EXAMPLE_DIR.rglob("*.yaml")) + list(EXAMPLE_DIR.rglob("*.yml"))
    return sorted(files)


YAML_FILES = _get_yaml_files()


class TestYamlLoadConfig:
    """Test cases for YAML configuration loading via load_config()."""

    @pytest.mark.skipif(not YAML_FILES, reason="No YAML files found")
    @pytest.mark.parametrize(
        "yaml_file",
        YAML_FILES,
        ids=lambda p: str(p.relative_to(EXAMPLE_DIR)),
    )
    def test_yaml_load_config(self, yaml_file: Path):
        """Test that each YAML file can be loaded via load_config()."""
        try:
            load_config(str(yaml_file))
        except Exception as e:
            pytest.fail(f"load_config failed for {yaml_file.name}: {e}")
