import os

# turn on the fast Rust-based downloader before any HF imports
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import argparse
from transformers import GPT2Model, GPT2Tokenizer


def download_gpt2_weights(model_size, models_dir):
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    model_name_map = {
        "124M": "gpt2",
        "355M": "gpt2-medium",
        "774M": "gpt2-large",
        "1558M": "gpt2-xl",
    }
    model_name = model_name_map[model_size]
    repo_id = f"openai-community/{model_name}"

    model_dir = os.path.join(models_dir, model_size)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Downloading GPT-2 ({model_size}) with hf_transfer …")
    model = GPT2Model.from_pretrained(repo_id)
    tokenizer = GPT2Tokenizer.from_pretrained(repo_id)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    print(f"Done. GPT-2 ({model_size}) weights saved in: {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download GPT-2 model weights without loading them."
    )
    parser.add_argument(
        "--model-size",
        default="124M",
        choices=["124M", "355M", "774M", "1558M"],
        help="GPT-2 model size to download (default: 124M)",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to store downloaded model weights (default: models)",
    )
    args = parser.parse_args()

    download_gpt2_weights(args.model_size, args.models_dir)
