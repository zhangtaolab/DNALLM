"""Test MCP server functionality without starting HTTP server."""

import asyncio
import sys
from pathlib import Path
from loguru import logger

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dnallm.mcp.server import DNALLMMCPServer  # noqa: E402


async def test_mcp_functionality():
    """Test MCP server functionality with the provided DNA sequence."""

    # Setup logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # DNA sequence to test
    dna_sequence = "GGGCAGCGGTTACACCTTAATCGACACGACTCTCGGCAACGGATATCTCGGCTCTTGCATCGATGAAGAACGTAGCAAAATGCGATACCTGGTGTGAATTGCAGAATCCCGCGAACCATCGAGTTTTTGAACGCAAGTTGCGCCCGAAGCCTTCTGACGGAGGGCACGTCTGCCTGGGCGTCACGCCAAAAGACACTCCCAACACCCCCCCGCGGGGCGAGGGACGTGGCGTCTGGCCCCCCGCGCTGCAGGGCGAGGTGGGCCGAAGCAGGGGCTGCCGGCGAACCGCGTCGGACGCAACACGTGGTGGGCGACATCAAGTTGTTCTCGGTGCAGCGTCCCGGCGCGCGGCCGGCCATTCGGCCCTAAGGACCCATCGAGCGACCGAGCTTGCCCTCGGACCACGACCCCAGGTCAGTCGGGACTACCCGCTGAGTTTAAGCATATAAATAAGCGGAGGAGAAGAAACTTACGAGGATTCCCCTAGTAACGGCGAGCGAACCGGGAGCAGCCCAGCTTGAGAATCGGGCGGCCTCGCCGCCCGAATTGTAGTCTGGAGAGGCGT"

    try:
        # Create server instance
        logger.info("Creating DNALLM MCP Server...")
        config_path = "./configs/mcp_server_config.yaml"
        server = DNALLMMCPServer(config_path)

        # Initialize server
        logger.info("Initializing server...")
        await server.initialize()

        # Get server info
        info = server.get_server_info()
        logger.info(f"Server initialized: {info['name']} v{info['version']}")
        logger.info(f"Loaded models: {info['loaded_models']}")
        logger.info(f"Enabled models: {info['enabled_models']}")

        # Test DNA sequence prediction
        logger.info(f"Testing DNA sequence (length: {len(dna_sequence)})")
        logger.info(f"Sequence: {dna_sequence[:50]}...")

        # Test single sequence prediction with promoter model
        logger.info("Testing promoter prediction...")
        promoter_result = await server.model_manager.predict_sequence(
            "promoter_model", dna_sequence
        )

        if promoter_result and 0 in promoter_result:
            result = promoter_result[0]
            logger.info("Promoter prediction result:")
            logger.info(f"  Label: {result.get('label', 'N/A')}")
            logger.info(f"  Scores: {result.get('scores', 'N/A')}")
            # Get confidence (max score)
            scores = result.get("scores", {})
            if scores:
                max_score = (
                    max(scores.values())
                    if isinstance(scores, dict)
                    else max(scores)
                )
                logger.info(f"  Confidence: {max_score:.4f}")
        else:
            logger.error("Promoter prediction failed")

        # Test conservation prediction
        logger.info("Testing conservation prediction...")
        conservation_result = await server.model_manager.predict_sequence(
            "conservation_model", dna_sequence
        )

        if conservation_result and 0 in conservation_result:
            result = conservation_result[0]
            logger.info("Conservation prediction result:")
            logger.info(f"  Label: {result.get('label', 'N/A')}")
            logger.info(f"  Scores: {result.get('scores', 'N/A')}")
            # Get confidence (max score)
            scores = result.get("scores", {})
            if scores:
                max_score = (
                    max(scores.values())
                    if isinstance(scores, dict)
                    else max(scores)
                )
                logger.info(f"  Confidence: {max_score:.4f}")
        else:
            logger.error("Conservation prediction failed")

        # Test open chromatin prediction
        logger.info("Testing open chromatin prediction...")
        chromatin_result = await server.model_manager.predict_sequence(
            "open_chromatin_model", dna_sequence
        )

        if chromatin_result and 0 in chromatin_result:
            result = chromatin_result[0]
            logger.info("Open chromatin prediction result:")
            logger.info(f"  Label: {result.get('label', 'N/A')}")
            logger.info(f"  Scores: {result.get('scores', 'N/A')}")
            # Get confidence (max score)
            scores = result.get("scores", {})
            if scores:
                max_score = (
                    max(scores.values())
                    if isinstance(scores, dict)
                    else max(scores)
                )
                logger.info(f"  Confidence: {max_score:.4f}")
        else:
            logger.error("Open chromatin prediction failed")

        # Summary
        logger.info("=" * 60)
        logger.info("PREDICTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"DNA Sequence: {dna_sequence[:50]}...")
        logger.info(f"Sequence Length: {len(dna_sequence)} bp")

        if promoter_result and 0 in promoter_result:
            result = promoter_result[0]
            scores = result.get("scores", {})
            max_score = max(scores.values()) if scores else 0
            logger.info(
                f"Promoter Prediction: {result.get('label', 'N/A')} (confidence: {max_score:.4f})"
            )
        if conservation_result and 0 in conservation_result:
            result = conservation_result[0]
            scores = result.get("scores", {})
            max_score = max(scores.values()) if scores else 0
            logger.info(
                f"Conservation Prediction: {result.get('label', 'N/A')} (confidence: {max_score:.4f})"
            )
        if chromatin_result and 0 in chromatin_result:
            result = chromatin_result[0]
            scores = result.get("scores", {})
            max_score = max(scores.values()) if scores else 0
            logger.info(
                f"Open Chromatin Prediction: {result.get('label', 'N/A')} (confidence: {max_score:.4f})"
            )

        # Shutdown server
        await server.shutdown()
        logger.info("Test completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_mcp_functionality())
