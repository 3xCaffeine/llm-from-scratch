"""Test script to verify GPT-2 weight loading works correctly."""

import sys
from gpt_model.load_weights import load_gpt2_params


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


if __name__ == "__main__":
    success = test_weight_loading()
    sys.exit(0 if success else 1)
