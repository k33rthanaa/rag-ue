
import argparse
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

from utils import setup_logging


def load_config(config_path: str | None = None) -> dict:
    """
    Load YAML config. Defaults to configs/default.yaml
    relative to the project root.
    """
    root = Path(__file__).resolve().parents[1]
    if config_path is None:
        config_path = root / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    
    # Setup logging
    root = Path(__file__).resolve().parents[1]
    log_dir = root / cfg.get("paths", {}).get("logs_dir", "outputs/logs")
    logger = setup_logging(str(log_dir), cfg.get("runtime", {}).get("log_level", "INFO"))

    dataset_cfg = cfg.get("dataset", {})
    repo_id = dataset_cfg.get("repo_id")
    filename = dataset_cfg.get("filename")

    if not repo_id or not filename:
        raise ValueError("Config missing dataset.repo_id or dataset.filename")

    # Store dataset under <project_root>/data/
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"⬇️ Downloading dataset from HF repo: {repo_id}")
    logger.info(f"   Filename: {filename}")
    logger.info(f"   Local directory: {data_dir}")

    dataset_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(data_dir),
    )

    logger.info("\n✅ Download complete.")
    logger.info(f"📁 Local dataset path: {dataset_path}")
    logger.info("\nNote: Other scripts should use this path when reading the dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset to local data/ folder.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    args = parser.parse_args()
    main(args)
