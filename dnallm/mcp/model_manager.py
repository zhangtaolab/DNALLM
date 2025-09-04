"""Model Manager for MCP Server.

This module provides model management functionality for the MCP server,
including model loading, caching, and prediction orchestration.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from loguru import logger as loguru_logger
import torch
import time

from dnallm.models import load_model_and_tokenizer
from dnallm.inference.predictor import DNAPredictor
from dnallm.configuration.configs import TaskConfig
from dnallm.utils import get_logger
from config_manager import MCPConfigManager
from config_validators import InferenceModelConfig

logger = get_logger("dnallm.mcp.model_manager")

class ModelManager:
    """Manages DNA prediction models and their lifecycle."""
    
    def __init__(self, config_manager: MCPConfigManager):
        """Initialize the model manager.
        
        Args:
            config_manager: MCPConfigManager instance
        """
        self.config_manager = config_manager
        self.loaded_models: Dict[str, DNAPredictor] = {}
        self.model_loading_status: Dict[str, str] = {}  # "loading", "loaded", "error"
        self._loading_lock = asyncio.Lock()
    
    async def load_model(self, model_name: str) -> bool:
        """Load a specific model asynchronously.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        async with self._loading_lock:
            if model_name in self.loaded_models:
                logger.info(f"Model {model_name} already loaded")
                return True
            
            if model_name in self.model_loading_status:
                if self.model_loading_status[model_name] == "loading":
                    logger.info(f"Model {model_name} is already being loaded")
                    return False
                elif self.model_loading_status[model_name] == "error":
                    logger.warning(f"Model {model_name} previously failed to load")
                    return False
            
            self.model_loading_status[model_name] = "loading"
            logger.info(f"Starting to load model: {model_name}")
            
            try:
                # Get model configuration
                model_config = self.config_manager.get_model_config(model_name)
                if not model_config:
                    raise ValueError(f"Configuration not found for model: {model_name}")
                
                # Display loading progress
                logger.progress(f"Loading model: {model_name}")
                logger.info(f"   Model path: {model_config.model.path}")
                logger.info(f"   Source: {model_config.model.source}")
                logger.info(f"   Task type: {model_config.task.task_type}")
                logger.info(f"   Architecture: {model_config.model.task_info.architecture}")
                logger.info(f"   📥 Downloading/loading model and tokenizer...")
                
                # Create task config for model loading
                task_config = TaskConfig(
                    task_type=model_config.task.task_type,
                    num_labels=model_config.task.num_labels,
                    label_names=model_config.task.label_names,
                    threshold=model_config.task.threshold
                )
                
                # Load model and tokenizer in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                start_time = time.time()
                model, tokenizer = await loop.run_in_executor(
                    None,
                    self._load_model_sync,
                    model_config.model.path,
                    task_config,
                    model_config.model.source
                )
                
                load_time = time.time() - start_time
                logger.success(f"Model and tokenizer loaded in {load_time:.2f} seconds")
                
                # Create predictor
                logger.info(f"   🔧 Creating DNA predictor...")
                predictor_config = {
                    'task': model_config.task,
                    'inference': model_config.inference
                }
                
                predictor = DNAPredictor(model, tokenizer, predictor_config)
                self.loaded_models[model_name] = predictor
                self.model_loading_status[model_name] = "loaded"
                
                total_time = time.time() - start_time
                logger.success(f"Successfully loaded model: {model_name} (total: {total_time:.2f}s)")
                loguru_logger.info(f"Successfully loaded model: {model_name}")
                return True
                
            except Exception as e:
                self.model_loading_status[model_name] = "error"
                logger.failure(f"Failed to load model {model_name}: {e}")
                loguru_logger.error(f"Failed to load model {model_name}: {e}")
                return False
    
    def _load_model_sync(self, model_path: str, task_config: TaskConfig, source: str) -> Tuple[Any, Any]:
        """Synchronously load model and tokenizer.
        
        Args:
            model_path: Path to the model
            task_config: Task configuration
            source: Model source (huggingface/modelscope)
            
        Returns:
            Tuple of (model, tokenizer)
        """
        return load_model_and_tokenizer(model_name=model_path, task_config=task_config, source=source)
    
    async def load_all_enabled_models(self) -> Dict[str, bool]:
        """Load all enabled models asynchronously.
        
        Returns:
            Dictionary mapping model names to loading success status
        """
        enabled_models = self.config_manager.get_enabled_models()
        logger.info(f"\n🚀 Starting to load {len(enabled_models)} enabled models:")
        for i, model_name in enumerate(enabled_models, 1):
            logger.info(f"   {i}. {model_name}")
        logger.info("")
        
        logger.info(f"Loading {len(enabled_models)} enabled models: {enabled_models}")
        
        # Load models concurrently
        tasks = [self.load_model(model_name) for model_name in enabled_models]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        loading_results = {}
        for i, model_name in enumerate(enabled_models):
            result = results[i]
            if isinstance(result, Exception):
                logger.error(f"Exception loading model {model_name}: {result}")
                loading_results[model_name] = False
            else:
                loading_results[model_name] = result
        
        successful_loads = sum(loading_results.values())
        logger.info(f"\n📊 Loading Summary:")
        logger.info(f"   ✅ Successfully loaded: {successful_loads}/{len(enabled_models)} models")
        logger.info(f"   ❌ Failed to load: {len(enabled_models) - successful_loads}/{len(enabled_models)} models")
        
        if successful_loads > 0:
            logger.info(f"\n🎉 Successfully loaded models:")
            for model_name, success in loading_results.items():
                if success:
                    logger.success(f"   {model_name}")
        
        if successful_loads < len(enabled_models):
            logger.warning_icon(f"Failed to load models:")
            for model_name, success in loading_results.items():
                if not success:
                    logger.failure(f"   {model_name}")
        
        logger.info(f"Successfully loaded {successful_loads}/{len(enabled_models)} models")
        
        return loading_results
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded model names.
        
        Returns:
            List of loaded model names
        """
        return list(self.loaded_models.keys())
    
    def get_model_status(self, model_name: str) -> str:
        """Get loading status of a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Status string: "loading", "loaded", "error", or "not_found"
        """
        return self.model_loading_status.get(model_name, "not_found")
    
    def get_predictor(self, model_name: str) -> Optional[DNAPredictor]:
        """Get predictor instance for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            DNAPredictor instance or None if not loaded
        """
        return self.loaded_models.get(model_name)
    
    async def predict_sequence(self, model_name: str, sequence: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Predict using a specific model.
        
        Args:
            model_name: Name of the model to use
            sequence: DNA sequence to predict
            **kwargs: Additional prediction parameters
            
        Returns:
            Prediction results or None if model not available
        """
        predictor = self.get_predictor(model_name)
        if not predictor:
            logger.error(f"Model {model_name} not loaded")
            return None
        
        try:
            # Run prediction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                predictor.predict_seqs,
                sequence,
                **kwargs
            )
            return result
        except Exception as e:
            logger.error(f"Prediction failed for model {model_name}: {e}")
            return None
    
    async def predict_batch(self, model_name: str, sequences: List[str], **kwargs) -> Optional[Dict[str, Any]]:
        """Predict using a specific model on a batch of sequences.
        
        Args:
            model_name: Name of the model to use
            sequences: List of DNA sequences to predict
            **kwargs: Additional prediction parameters
            
        Returns:
            Batch prediction results or None if model not available
        """
        predictor = self.get_predictor(model_name)
        if not predictor:
            logger.error(f"Model {model_name} not loaded")
            return None
        
        try:
            # Run prediction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                predictor.predict_seqs,
                sequences,
                **kwargs
            )
            return result
        except Exception as e:
            logger.error(f"Batch prediction failed for model {model_name}: {e}")
            return None
    
    async def predict_multi_model(self, model_names: List[str], sequence: str, **kwargs) -> Dict[str, Any]:
        """Predict using multiple models in parallel.
        
        Args:
            model_names: List of model names to use
            sequence: DNA sequence to predict
            **kwargs: Additional prediction parameters
            
        Returns:
            Dictionary mapping model names to prediction results
        """
        logger.info(f"Running multi-model prediction with {len(model_names)} models")
        
        # Create prediction tasks
        tasks = [
            self.predict_sequence(model_name, sequence, **kwargs)
            for model_name in model_names
        ]
        
        # Run predictions concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        multi_results = {}
        for i, model_name in enumerate(model_names):
            result = results[i]
            if isinstance(result, Exception):
                logger.error(f"Exception in multi-model prediction for {model_name}: {result}")
                multi_results[model_name] = {"error": str(result)}
            else:
                multi_results[model_name] = result
        
        return multi_results
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary or None if not found
        """
        model_config = self.config_manager.get_model_config(model_name)
        if not model_config:
            return None
        
        predictor = self.get_predictor(model_name)
        
        info = {
            "name": model_name,
            "task_type": model_config.task.task_type,
            "num_labels": model_config.task.num_labels,
            "label_names": model_config.task.label_names,
            "model_path": model_config.model.path,
            "model_source": model_config.model.source,
            "architecture": model_config.model.task_info.architecture,
            "tokenizer": model_config.model.task_info.tokenizer,
            "species": model_config.model.task_info.species,
            "task_category": model_config.model.task_info.task_category,
            "performance_metrics": model_config.model.task_info.performance_metrics,
            "status": self.get_model_status(model_name),
            "loaded": model_name in self.loaded_models
        }
        
        if predictor:
            try:
                memory_usage = predictor.estimate_memory_usage()
                info["memory_usage"] = memory_usage
            except Exception as e:
                logger.warning(f"Could not estimate memory usage for {model_name}: {e}")
        
        return info
    
    def get_all_models_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all configured models.
        
        Returns:
            Dictionary mapping model names to their information
        """
        all_models = {}
        for model_name in self.config_manager.get_enabled_models():
            info = self.get_model_info(model_name)
            if info:
                all_models[model_name] = info
        
        return all_models
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a specific model to free memory.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            True if model was unloaded, False if not found
        """
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            if model_name in self.model_loading_status:
                del self.model_loading_status[model_name]
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Unloaded model: {model_name}")
            return True
        
        return False
    
    def unload_all_models(self) -> int:
        """Unload all loaded models.
        
        Returns:
            Number of models unloaded
        """
        unloaded_count = 0
        for model_name in list(self.loaded_models.keys()):
            if self.unload_model(model_name):
                unloaded_count += 1
        
        logger.info(f"Unloaded {unloaded_count} models")
        return unloaded_count
