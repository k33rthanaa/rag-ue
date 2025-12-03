import yaml
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM

# 1) CONFIG

def load_config(config_path: str | None = None):
    """
    Load YAML config. Defaults to configs/default.yaml
    relative to project root.
    """
    root = Path(__file__).resolve().parents[1]
    if config_path is None:
        config_path = root / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# 2) DEVICE + MODEL

def get_device():
    """Return cuda if available, else cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(cfg, task_type="retriever"):
    """
    Load tokenizer + model from config.
    task_type: "retriever" or "answering" (for retrieval or answering).
    """
    if task_type == "retriever":
        # Load Contriever model for retrieval
        model_name = cfg.get("model_name", "facebook/contriever")  # Contriever for retrieval
    elif task_type == "answering":
        # Load Qwen2.5-7B-Instruct model for answering
        model_name = cfg.get("answering_model_name", "Qwen/Qwen2.5-7B-Instruct")  # Qwen2.5 for answering
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Use 'retriever' or 'answering'.")

    use_fp16 = bool(cfg.get("use_fp16", False))

    device = get_device()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_kwargs = {}
    if use_fp16 and device.type == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    # Load the model
    if task_type == "retriever":
        model = AutoModel.from_pretrained(model_name, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    model.to(device)
    model.eval()

    return tokenizer, model, device



# 3) ENCODING

@torch.no_grad()
def mean_pooling(token_embeddings, attention_mask):
    """
    Mean pooling over non-padding tokens.
    token_embeddings: [B, T, H]
    attention_mask  : [B, T]
    """
    mask = attention_mask.unsqueeze(-1).float()
    masked = token_embeddings * mask
    summed = masked.sum(dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@torch.no_grad()
def encode_batch(texts, tokenizer, model, device, max_length=512):
    """
    Encode a list of texts into a numpy array [B, H] (float32).
    """
    if not texts:
        hidden = model.config.hidden_size
        return np.empty((0, hidden), dtype=np.float32)

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    outputs = model(**inputs)
    embeddings = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
    return embeddings.cpu().numpy().astype("float32")


# 4) LOGGING

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




