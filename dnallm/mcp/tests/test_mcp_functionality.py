"""Test MCP server functionality without starting HTTP server."""

import sys
from pathlib import Path
from loguru import logger
import pytest

from ..server import DNALLMMCPServer


def _assert_prediction(result_map, model_key, num_labels):
    """Assert an MCP prediction result is present and well-formed.

    Args:
        result_map: Value returned by ModelManager.predict_sequence — a dict
            keyed by sequence index on success, or None when the model failed
            to load or the prediction raised (ModelManager swallows both by
            design and never raises).
        model_key: Configured model name, used in failure messages.
        num_labels: Expected number of label-score entries for the task.

    Returns:
        The single per-sequence prediction dict (label/scores/sequence).
    """
    assert result_map, f"{model_key} prediction returned no result (model load or predict failed)"
    result = next(iter(result_map.values()))
    assert isinstance(result.get("label"), str), f"malformed {model_key} result: {result}"
    scores = result.get("scores")
    assert isinstance(scores, dict), f"malformed {model_key} scores: {result}"
    assert len(scores) == num_labels, (
        f"malformed {model_key} scores (expected {num_labels} entries): {result}"
    )
    return result


class TestMCPFunctionality:
    """Test MCP server functionality with DNA sequences."""

    @pytest.fixture(autouse=True)
    def setup_logging(self):
        """Setup logging for tests."""
        logger.remove()
        logger.add(
            sys.stderr,
            level="INFO",
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:"
                "<cyan>{line}</cyan> - <level>{message}</level>"
            ),
        )

    @pytest.fixture
    def dna_sequence(self):
        """Test DNA sequence."""
        return (
            "AGAAAAAACATGACAAGAAATCGATAATAATACAAAAGCTATGATGGTGTGCAATGTCCGT"
            "GTGCATGCGTGCACGCATTGCAACCGGCCCAAATCAAGGCCCATCGATCAGTGAATACTC"
            "ATGGGCCGGCGGCCCACCACCGCTTCATCTCCTCCTCCGACGACGGGAGCACCCCCGCCG"
            "CATCGCCACCGACGAGGAGGAGGCCATTGCCGGCGGCGCCCCCGGTGAGCCGCTGCACCA"
            "CGTCCCTGA"
        )

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.timeout(3600)
    async def test_mcp_functionality(self, dna_sequence):
        """Test MCP server functionality with the provided DNA sequence."""
        try:
            # Create server instance
            logger.info("Creating DNALLM MCP Server...")
            # Use absolute path to config file in tests directory
            config_path = Path(__file__).parent / "configs" / "mcp_server_config.yaml"
            server = DNALLMMCPServer(str(config_path))

            # Initialize server
            logger.info("Initializing server...")
            await server.initialize()

            # Get server info
            info = server.get_server_info()
            logger.info(f"Server initialized: {info['name']} v{info['version']}")
            logger.info(f"Loaded models: {info['loaded_models']}")
            logger.info(f"Enabled models: {info['enabled_models']}")

            # Model loads must actually succeed: initialize() gathers load
            # errors with return_exceptions=True and ModelManager.predict_sequence
            # returns None (never raises) for a not-loaded model, so without
            # these asserts a total load failure would still pass the test.
            manager = server.model_manager
            assert manager.loaded_models, (
                f"no models loaded after initialize: {manager.model_loading_status}"
            )

            # Test DNA sequence prediction
            logger.info(f"Testing DNA sequence (length: {len(dna_sequence)})")
            logger.info(f"Sequence: {dna_sequence[:50]}...")

            # Test single sequence prediction with promoter model (binary, 2 labels)
            logger.info("Testing promoter prediction...")
            promoter_result = await manager.predict_sequence("promoter_model", dna_sequence)
            promoter_pred = _assert_prediction(promoter_result, "promoter_model", num_labels=2)
            logger.info("Promoter prediction result:")
            logger.info(f"  Label: {promoter_pred['label']}")
            logger.info(f"  Scores: {promoter_pred['scores']}")
            logger.info(f"  Confidence: {max(promoter_pred['scores'].values()):.4f}")

            # Test conservation prediction (binary, 2 labels)
            logger.info("Testing conservation prediction...")
            conservation_result = await manager.predict_sequence("conservation_model", dna_sequence)
            conservation_pred = _assert_prediction(
                conservation_result, "conservation_model", num_labels=2
            )
            logger.info("Conservation prediction result:")
            logger.info(f"  Label: {conservation_pred['label']}")
            logger.info(f"  Scores: {conservation_pred['scores']}")
            logger.info(f"  Confidence: {max(conservation_pred['scores'].values()):.4f}")

            # Test open chromatin prediction (multiclass, 3 labels)
            logger.info("Testing open chromatin prediction...")
            chromatin_result = await manager.predict_sequence("open_chromatin_model", dna_sequence)
            chromatin_pred = _assert_prediction(
                chromatin_result, "open_chromatin_model", num_labels=3
            )
            logger.info("Open chromatin prediction result:")
            logger.info(f"  Label: {chromatin_pred['label']}")
            logger.info(f"  Scores: {chromatin_pred['scores']}")
            logger.info(f"  Confidence: {max(chromatin_pred['scores'].values()):.4f}")

            # Summary (results already validated above — unconditional)
            logger.info("=" * 60)
            logger.info("PREDICTION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"DNA Sequence: {dna_sequence[:50]}...")
            logger.info(f"Sequence Length: {len(dna_sequence)} bp")
            logger.info(
                f"Promoter Prediction: {promoter_pred['label']} "
                f"(confidence: {max(promoter_pred['scores'].values()):.4f})"
            )
            logger.info(
                f"Conservation Prediction: {conservation_pred['label']} "
                f"(confidence: {max(conservation_pred['scores'].values()):.4f})"
            )
            logger.info(
                f"Open Chromatin Prediction: {chromatin_pred['label']} "
                f"(confidence: {max(chromatin_pred['scores'].values()):.4f})"
            )

            # Shutdown server
            await server.shutdown()
            logger.info("Test completed successfully!")

        except Exception as e:
            logger.error(f"Test failed: {e}")
            import traceback

            traceback.print_exc()
            raise


if __name__ == "__main__":
    pytest.main([__file__])
