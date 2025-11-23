#!/usr/bin/env python3
"""
Shared utility functions for RAG Uncertainty Estimator.
"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np


def load_tokenizer(model_name: str):
    """Load tokenizer for a given model."""
    return AutoTokenizer.from_pretrained(model_name)


def get_encoder(model_name: str, device: str = None):
    """
    Get encoder model for generating embeddings.
    
    Args:
        model_name: Name of the model (e.g., "facebook/contriever")
        device: Device to use ("cuda" or "cpu"). If None, auto-detect.
    
    Returns:
        Encoder object with encode() method
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    class Encoder:
        def __init__(self, model, tokenizer, device):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
        
        def encode(self, text: str, normalize: bool = True):
            """Encode text into embedding vector."""
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling for Contriever
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
            
            if normalize:
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            return embeddings.cpu().numpy()
    
    return Encoder(model, tokenizer, device)


def load_config(config_path: str):
    """Load YAML configuration file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: str, level: str = "INFO"):
    """Setup logging configuration."""
    import logging
    import os
    from datetime import datetime
    
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(
        log_dir,
        f"rag_ue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

