import yaml
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig

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
    # Resolve potential local paths relative to project root
    root = Path(__file__).resolve().parents[1]

    if task_type == "retriever":
        # Load Contriever model for retrieval
        model_name = cfg.get("model_name", "facebook/contriever")  # Contriever for retrieval
    elif task_type == "answering":
        # Load Qwen2.5-7B-Instruct model for answering
        model_name = cfg.get("answering_model_name", "Qwen/Qwen2.5-7B-Instruct")  # Qwen2.5 for answering
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Use 'retriever' or 'answering'.")

    # If config provides a relative directory path (e.g. 'scripts/models/contriever'),
    # interpret it relative to the project root so it works no matter the CWD.
    candidate_dir = root / model_name
    if candidate_dir.exists():
        model_name = str(candidate_dir)

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

def load_model_and_tokenizer2(cfg, task_type="retriever"):
    """
    Load tokenizer + model from config.
    task_type: "retriever" (CPU) or "answering" (GPU with INT4)
    """
    # Resolve potential local paths relative to project root
    root = Path(__file__).resolve().parents[1]

    if task_type == "retriever":
        # Contriever - Always CPU
        model_name = cfg.get("model_name", "facebook/contriever")
        candidate_dir = root / model_name
        if candidate_dir.exists():
            model_name = str(candidate_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        use_fp16 = bool(cfg.get("use_fp16", False))
        device = get_device()
        
        model_kwargs = {}
        if use_fp16 and device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        
        model = AutoModel.from_pretrained(model_name, **model_kwargs)
        model.to(device)
        model.eval()
        
        return tokenizer, model, device
        
    elif task_type == "answering":
        # Qwen - GPU with INT4 quantization
        model_name = cfg.get("answering_model_name", "Qwen/Qwen2.5-7B-Instruct")
        candidate_dir = root / model_name
        if candidate_dir.exists():
            model_name = str(candidate_dir)
        
        # Check GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available. Qwen requires a GPU for answering.")
        
        print(f"Loading answering model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure INT4 quantization
        print("Configuring INT4 quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Set max memory limits
        max_memory = {0: "10GB", "cpu": "30GB"}
        
        # Load model with INT4
        print("Loading model with INT4 quantization (this can take several minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True
        )
        
        device = torch.device("cuda:0")
        print("Answering model loaded successfully with INT4.")
        
        return tokenizer, model, device
    
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Use 'retriever' or 'answering'.")



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




