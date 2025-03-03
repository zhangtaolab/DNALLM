import pytest
from click.testing import CliRunner
from dnallm.cli.train import main

def test_train_cli_help():
    """Test CLI help message"""
    runner = CliRunner()
    result = runner.invoke(main, ['--help'])
    assert result.exit_code == 0
    assert 'Fine-tune a DNA Language Model' in result.output

def test_train_cli_execution(sample_fasta_file, temp_dir):
    """Test CLI execution with minimal arguments"""
    runner = CliRunner()
    result = runner.invoke(main, [
        '--model-type', 'plant_dna',
        '--model-name', 'zhangtaolab/plant-dnabert-BPE',
        '--train-file', sample_fasta_file,
        '--eval-file', sample_fasta_file,
        '--output-dir', temp_dir
    ])
    assert result.exit_code == 0 