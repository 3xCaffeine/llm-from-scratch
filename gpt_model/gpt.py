# standalone script

import time
import tiktoken
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out is indivisible by num_heads"
        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out
        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

        # KV cache buffers
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0

    def forward(self, x, use_cache=False):
        batch_size, num_tokens, embed_dim = x.shape
        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)
        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys_new, values_new = qkv

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=2)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=2)
            keys, values = self.cache_k, self.cache_v
            self.ptr_current_pos += num_tokens
        else:
            keys, values = keys_new, values_new

        use_dropout = 0.0 if not self.training else self.dropout
        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True
        )
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = (
            context_vec.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_tokens, self.d_out)
        )
        context_vec = self.proj(context_vec)
        return context_vec

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)

        # x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]

        x = self.att(x, use_cache=use_cache)

        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # self.trf_blocks = nn.Sequential(
        #    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.current_pos = 0

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, use_cache=False):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        if use_cache:
            pos_ids = torch.arange(
                self.current_pos,
                self.current_pos + seq_len,
                device=in_idx.device,
                dtype=torch.long,
            )
            self.current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)

        # x = self.trf_blocks(x)

        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0

    @classmethod
    def from_pretrained(cls, model_name):
        from transformers import GPT2Model
        
        # Map model name to size
        model_size_map = {
            "gpt2": "124M",
            "gpt2-medium": "355M",
            "gpt2-large": "774M",
            "gpt2-xl": "1558M",
        }
        if model_name not in model_size_map:
            raise ValueError(f"Model name not in {list(model_size_map.keys())}")
        
        model_size = model_size_map[model_name]
        
        BASE_CONFIG = {
            "vocab_size": 50257,  # Vocabulary size
            "context_length": 1024,  # Context length
            "drop_rate": 0.0,  # Dropout rate
            "qkv_bias": True,  # Query-key-value bias
        }

        model_configs = {
            "124M": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "355M": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "774M": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "1558M": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        BASE_CONFIG.update(model_configs[model_size])
        
        model = cls(BASE_CONFIG)
        
        # Load from HF
        hf_model = GPT2Model.from_pretrained(model_name)
        state_dict = hf_model.state_dict()
        
        # Load weights into model
        from .load_weights import load_weights_into_gpt
        
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
            params["blocks"][b]["ln_1"]["g"] = state_dict[
                f"h.{b}.ln_1.weight"
            ].numpy()
            params["blocks"][b]["ln_1"]["b"] = state_dict[
                f"h.{b}.ln_1.bias"
            ].numpy()
            params["blocks"][b]["ln_2"] = {}
            params["blocks"][b]["ln_2"]["g"] = state_dict[
                f"h.{b}.ln_2.weight"
            ].numpy()
            params["blocks"][b]["ln_2"]["b"] = state_dict[
                f"h.{b}.ln_2.bias"
            ].numpy()

        params["g"] = state_dict["ln_f.weight"].numpy()
        params["b"] = state_dict["ln_f.bias"].numpy()

        load_weights_into_gpt(model, params)
        return model
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def generate_text_simple_cached(
    model, idx, max_new_tokens, context_size=None, use_cache=True
):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx


def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False,  # Query-Key-Value bias
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)

    print(f"\n{50 * '='}\n{22 * ' '}IN\n{50 * '='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    # token_ids = generate_text_simple(
    #     model=model,
    #     idx=encoded_tensor,
    #     max_new_tokens=200,
    #     context_size=GPT_CONFIG_124M["context_length"]
    # )

    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=200,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print(f"\n\n{50 * '='}\n{22 * ' '}OUT\n{50 * '='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0]) / total_time)} tokens/sec")
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024**3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
