import numpy as np
import torch
import os
from gpt_model import GPTModel


def load_gpt2_params(model_size, models_dir="gpt2-weights"):
    # Map model size to Hugging Face model name
    model_name_map = {
        "124M": "gpt2",
        "355M": "gpt2-medium",
        "774M": "gpt2-large",
        "1558M": "gpt2-xl",
    }
    if model_size not in model_name_map:
        raise ValueError(f"Model size not in {list(model_name_map.keys())}")

    model_dir = os.path.join(models_dir, model_size)
    params_path = os.path.join(model_dir, "params.pt")

    if os.path.exists(params_path):
        print(f"Loading existing params from {params_path}")
        # Add numpy to safe globals for PyTorch 2.6+ compatibility
        torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
        params = torch.load(params_path, weights_only=False)
    else:
        print(f"Loading params from Hugging Face for {model_size}")
        model_name = model_name_map[model_size]
        hf_model = GPTModel.from_pretrained(model_name)
        state_dict = hf_model.state_dict()

        # Create params dict in the expected format
        params = {"blocks": [{} for _ in range(hf_model.config.n_layer)]}

        params["wpe"] = state_dict["wpe.weight"].numpy()
        params["wte"] = state_dict["wte.weight"].numpy()

        for b in range(hf_model.config.n_layer):
            params["blocks"][b]["attn"] = {}
            params["blocks"][b]["attn"]["c_attn"] = {}
            params["blocks"][b]["attn"]["c_attn"]["w"] = state_dict[
                f"h.{b}.attn.c_attn.weight"
            ].numpy()
            params["blocks"][b]["attn"]["c_attn"]["b"] = state_dict[
                f"h.{b}.attn.c_attn.bias"
            ].numpy()
            params["blocks"][b]["attn"]["c_proj"] = {}
            params["blocks"][b]["attn"]["c_proj"]["w"] = state_dict[
                f"h.{b}.attn.c_proj.weight"
            ].numpy()
            params["blocks"][b]["attn"]["c_proj"]["b"] = state_dict[
                f"h.{b}.attn.c_proj.bias"
            ].numpy()

            params["blocks"][b]["mlp"] = {}
            params["blocks"][b]["mlp"]["c_fc"] = {}
            params["blocks"][b]["mlp"]["c_fc"]["w"] = state_dict[
                f"h.{b}.mlp.c_fc.weight"
            ].numpy()
            params["blocks"][b]["mlp"]["c_fc"]["b"] = state_dict[
                f"h.{b}.mlp.c_fc.bias"
            ].numpy()
            params["blocks"][b]["mlp"]["c_proj"] = {}
            params["blocks"][b]["mlp"]["c_proj"]["w"] = state_dict[
                f"h.{b}.mlp.c_proj.weight"
            ].numpy()
            params["blocks"][b]["mlp"]["c_proj"]["b"] = state_dict[
                f"h.{b}.mlp.c_proj.bias"
            ].numpy()

            params["blocks"][b]["ln_1"] = {}
            params["blocks"][b]["ln_1"]["g"] = state_dict[f"h.{b}.ln_1.weight"].numpy()
            params["blocks"][b]["ln_1"]["b"] = state_dict[f"h.{b}.ln_1.bias"].numpy()
            params["blocks"][b]["ln_2"] = {}
            params["blocks"][b]["ln_2"]["g"] = state_dict[f"h.{b}.ln_2.weight"].numpy()
            params["blocks"][b]["ln_2"]["b"] = state_dict[f"h.{b}.ln_2.bias"].numpy()

        params["g"] = state_dict["ln_f.weight"].numpy()
        params["b"] = state_dict["ln_f.bias"].numpy()

        # Save to local dir
        os.makedirs(model_dir, exist_ok=True)
        torch.save(params, params_path)
        print(f"Saved params to {params_path}")

    return params


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    # Create tensor on same device and dtype as left parameter
    return torch.nn.Parameter(
        torch.as_tensor(right, dtype=left.dtype, device=left.device)
    )


def load_weights_into_gpt(gpt, params):
    """Load GPT-2 weights into model. Auto-detects architecture (combined qkv vs separate)."""
    # Detect if model uses combined qkv or separate W_query, W_key, W_value
    has_combined_qkv = hasattr(gpt.trf_blocks[0].att, "qkv")

    if has_combined_qkv:
        _load_weights_combined_qkv(gpt, params)
    else:
        _load_weights_separate_qkv(gpt, params)


def _load_weights_combined_qkv(gpt, params):
    """Load weights for models with combined qkv linear layer (like gpt.py)."""
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        # Combined qkv weight and bias
        qkv_w = params["blocks"][b]["attn"]["c_attn"]["w"]  # (emb_dim, 3*emb_dim)
        gpt.trf_blocks[b].att.qkv.weight = assign(
            gpt.trf_blocks[b].att.qkv.weight, qkv_w.T
        )
        qkv_b = params["blocks"][b]["attn"]["c_attn"]["b"]  # (3*emb_dim,)
        gpt.trf_blocks[b].att.qkv.bias = assign(gpt.trf_blocks[b].att.qkv.bias, qkv_b)

        # Output projection
        gpt.trf_blocks[b].att.proj.weight = assign(
            gpt.trf_blocks[b].att.proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].att.proj.bias = assign(
            gpt.trf_blocks[b].att.proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        # MLP layers
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        # Layer norms
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
        )

    # Final layer norm and output head
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def _load_weights_separate_qkv(gpt, params):
    """Load weights for models with separate W_query, W_key, W_value (like gpt_swa.py, gpt_gqa.py)."""
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T
        )

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b
        )

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
        )

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())
