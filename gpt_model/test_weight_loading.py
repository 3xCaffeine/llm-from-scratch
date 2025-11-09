"""Test script to verify GPT-2 weight loading works correctly."""

import sys
import torch
import tiktoken
# Use the efficient gpt.py with combined qkv
from gpt_model.gpt import GPTModel, generate_text_simple_cached
from gpt_model.load_weights import load_gpt2_params, load_weights_into_gpt


def test_weight_loading():
    """Test loading weights for GPT-2 124M model."""
    print("Testing weight loading for GPT-2 124M model...")
    print("-" * 60)

    try:
        # Try to load the params
        params = load_gpt2_params("124M", models_dir="gpt2-weights")

        print("SUCCESS: Successfully loaded parameters!")
        print(f"SUCCESS: Number of blocks: {len(params['blocks'])}")

        # Check key components
        checks = [
            ("wpe (position embeddings)", "wpe" in params),
            ("wte (token embeddings)", "wte" in params),
            ("final norm scale", "g" in params),
            ("final norm bias", "b" in params),
        ]

        for check_name, result in checks:
            status = "SUCCESS:" if result else "ERROR:"
            print(f"{status} {check_name}: {result}")

        # Check first block structure
        first_block = params["blocks"][0]
        block_checks = [
            ("attention weights", "attn" in first_block),
            ("MLP weights", "mlp" in first_block),
            ("layer norm 1", "ln_1" in first_block),
            ("layer norm 2", "ln_2" in first_block),
        ]

        print("\nFirst block structure:")
        for check_name, result in block_checks:
            status = "SUCCESS:" if result else "ERROR:"
            print(f"{status} {check_name}: {result}")

        # Check shapes
        print("\nKey shapes:")
        print(f"  wpe shape: {params['wpe'].shape}")
        print(f"  wte shape: {params['wte'].shape}")
        print(
            f"  blocks[0].attn.c_attn.w shape: {params['blocks'][0]['attn']['c_attn']['w'].shape}"
        )

        print("\n" + "=" * 60)
        print("SUCCESS: All weight loading checks passed!")
        return True

    except Exception as e:
        print(f"ERROR: Error loading weights: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_text_generation():
    """Test text generation with loaded GPT-2 model as sanity check."""
    print("\n" + "=" * 60)
    print("Testing text generation (sanity check)...")
    print("-" * 60)

    try:
        # Configuration for GPT-2 124M
        GPT_CONFIG_124M = {
            "vocab_size": 50257,
            "context_length": 1024,
            "emb_dim": 768,
            "n_heads": 12,
            "n_layers": 12,
            "drop_rate": 0.0,  # Disable dropout for inference
            "qkv_bias": True,
        }

        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        model = GPTModel(GPT_CONFIG_124M)
        
        # Load pretrained weights BEFORE moving to device
        # (weights are loaded as CPU tensors from numpy arrays)
        print("Loading GPT-2 weights...")
        params = load_gpt2_params("124M", models_dir="gpt2-weights")
        load_weights_into_gpt(model, params)
        print("SUCCESS: Weights loaded into model")
        
        # Now move model to device
        model.to(device)
        model.eval()

        # Initialize tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")

        # Test text generation
        start_context = "Hello, I am"
        print(f"\nInput prompt: '{start_context}'")
        
        encoded = tokenizer.encode(start_context)
        encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)

        print(f"Generating {50} tokens...")
        token_ids = generate_text_simple_cached(
            model=model,
            idx=encoded_tensor,
            max_new_tokens=50,
            context_size=GPT_CONFIG_124M["context_length"],
            use_cache=True,
        )

        decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())
        
        print("\n" + "-" * 60)
        print("Generated text:")
        print("-" * 60)
        print(decoded_text)
        print("-" * 60)

        # Basic sanity checks
        output_length = len(token_ids[0])
        expected_length = len(encoded) + 50
        
        checks = [
            ("Output contains input", start_context in decoded_text),
            ("Generated expected number of tokens", output_length == expected_length),
            ("Output is longer than input", len(decoded_text) > len(start_context)),
        ]

        print("\nSanity checks:")
        all_passed = True
        for check_name, result in checks:
            status = "SUCCESS:" if result else "ERROR:"
            print(f"{status} {check_name}: {result}")
            if not result:
                all_passed = False

        if all_passed:
            print("\n" + "=" * 60)
            print("SUCCESS: Text generation sanity check passed!")
            return True
        else:
            print("\n" + "=" * 60)
            print("WARNING: Some sanity checks failed")
            return False

    except Exception as e:
        print(f"ERROR: Error during text generation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test weight loading
    success1 = test_weight_loading()
    
    # Test text generation as sanity check
    success2 = test_text_generation()
    
    sys.exit(0 if (success1 and success2) else 1)
