"""Test MCP server functionality without starting HTTP server."""

import asyncio
import sys
from pathlib import Path
from loguru import logger

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dnallm.mcp.server import DNALLMMCPServer


async def test_mcp_functionality():
    """Test MCP server functionality with the provided DNA sequence."""
    
    # Setup logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
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
        promoter_result = await server.model_manager.predict_sequence("promoter_model", dna_sequence)
        
        if promoter_result:
            logger.info("Promoter prediction result:")
            logger.info(f"  Prediction: {promoter_result.get('prediction', 'N/A')}")
            logger.info(f"  Confidence: {promoter_result.get('confidence', 'N/A')}")
            logger.info(f"  Label: {promoter_result.get('label', 'N/A')}")
        else:
            logger.error("Promoter prediction failed")
        
        # Test conservation prediction
        logger.info("Testing conservation prediction...")
        conservation_result = await server.model_manager.predict_sequence("conservation_model", dna_sequence)
        
        if conservation_result:
            logger.info("Conservation prediction result:")
            logger.info(f"  Prediction: {conservation_result.get('prediction', 'N/A')}")
            logger.info(f"  Confidence: {conservation_result.get('confidence', 'N/A')}")
            logger.info(f"  Label: {conservation_result.get('label', 'N/A')}")
        else:
            logger.error("Conservation prediction failed")
        
        # Test open chromatin prediction
        logger.info("Testing open chromatin prediction...")
        chromatin_result = await server.model_manager.predict_sequence("open_chromatin_model", dna_sequence)
        
        if chromatin_result:
            logger.info("Open chromatin prediction result:")
            logger.info(f"  Prediction: {chromatin_result.get('prediction', 'N/A')}")
            logger.info(f"  Confidence: {chromatin_result.get('confidence', 'N/A')}")
            logger.info(f"  Label: {chromatin_result.get('label', 'N/A')}")
        else:
            logger.error("Open chromatin prediction failed")
        
        # Summary
        logger.info("=" * 60)
        logger.info("PREDICTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"DNA Sequence: {dna_sequence[:50]}...")
        logger.info(f"Sequence Length: {len(dna_sequence)} bp")
        
        if promoter_result:
            logger.info(f"Promoter Prediction: {promoter_result.get('label', 'N/A')} (confidence: {promoter_result.get('confidence', 'N/A')})")
        if conservation_result:
            logger.info(f"Conservation Prediction: {conservation_result.get('label', 'N/A')} (confidence: {conservation_result.get('confidence', 'N/A')})")
        if chromatin_result:
            logger.info(f"Open Chromatin Prediction: {chromatin_result.get('label', 'N/A')} (confidence: {chromatin_result.get('confidence', 'N/A')})")
        
        # Shutdown server
        await server.shutdown()
        logger.info("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_mcp_functionality())
