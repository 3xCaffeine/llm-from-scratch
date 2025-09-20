import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    from importlib.metadata import version
    import torch.nn as nn
    print("torch version:", version("torch"))
    return mo, nn, torch


@app.cell
def _(mo):
    mo.md(r"""simple self-attention mechanism without trainable weights""")
    return


@app.cell
def _(torch):
    _inputs = torch.tensor(
      [[0.43, 0.15, 0.89], # Your     (x^1)
       [0.55, 0.87, 0.66], # journey  (x^2)
       [0.57, 0.85, 0.64], # starts   (x^3)
       [0.22, 0.58, 0.33], # with     (x^4)
       [0.77, 0.25, 0.10], # one      (x^5)
       [0.05, 0.80, 0.55]] # step     (x^6)
    )

    query = _inputs[1]  # 2nd input token is the query

    attn_scores_2 = torch.empty(_inputs.shape[0])
    for _i, x_i in enumerate(_inputs):
        attn_scores_2[_i] = torch.dot(x_i, query) # dot product (transpose not necessary here since they are 1-dim vectors)

    print(attn_scores_2)

    res = 0.

    for idx, element in enumerate(_inputs[0]):
        res += _inputs[0][idx] * query[idx]

    print(res)
    print(torch.dot(_inputs[0], query))

    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

    print("Attention weights:", attn_weights_2)
    print("Sum:", attn_weights_2.sum())

    query = _inputs[1] # 2nd input token is the query

    context_vec_2 = torch.zeros(query.shape)
    for i,x_i in enumerate(_inputs):
        context_vec_2 += attn_weights_2[i]*x_i

    print(context_vec_2)

    inputs=_inputs
    return context_vec_2, inputs


@app.cell
def _(mo):
    mo.md(r"""attn weights for all input tokens""")
    return


@app.cell
def _(inputs, torch):
    _attn_scores = torch.empty(6, 6)

    for _i, _x_i in enumerate(inputs):
        for _j, _x_j in enumerate(inputs):
            _attn_scores[_i, _j] = torch.dot(_x_i, _x_j)

    print(_attn_scores)
    return


@app.cell
def _(inputs):
    attn_scores = inputs @ inputs.T
    print(attn_scores)
    return (attn_scores,)


@app.cell
def _(attn_scores, torch):
    attn_weights = torch.softmax(attn_scores, dim=-1)
    print(attn_weights)
    return (attn_weights,)


@app.cell
def _(attn_weights):
    row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
    print("Row 2 sum:", row_2_sum)

    print("All row sums:", attn_weights.sum(dim=-1))
    return


@app.cell
def _(attn_weights, inputs):
    all_context_vecs = attn_weights @ inputs
    print(all_context_vecs)
    return


@app.cell
def _(context_vec_2):
    print("Previous 2nd context vector:", context_vec_2)
    return


@app.cell
def _(mo):
    mo.md(r"""impl self-attention with trainable weights""")
    return


@app.cell
def _(inputs):
    x_2 = inputs[1] # second input element
    d_in = inputs.shape[1] # the input embedding size, d=3
    d_out = 2 # the output embedding size, d=2
    return d_in, d_out, x_2


@app.cell
def _(d_in, d_out, torch):
    torch.manual_seed(123)

    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    return W_key, W_query, W_value


@app.cell
def _(W_key, W_query, W_value, x_2):
    query_2 = x_2 @ W_query # _2 because it's with respect to the 2nd input element
    key_2 = x_2 @ W_key 
    value_2 = x_2 @ W_value

    print(query_2)
    return (query_2,)


@app.cell
def _(W_key, W_value, inputs):
    keys = inputs @ W_key 
    values = inputs @ W_value

    print("keys.shape:", keys.shape)
    print("values.shape:", values.shape)
    return keys, values


@app.cell
def _(keys, query_2):
    keys_2 = keys[1] # Python starts index at 0
    attn_score_22 = query_2.dot(keys_2)
    print(attn_score_22)
    return


@app.cell
def _(keys, query_2):
    attn_score_2 = query_2 @ keys.T # All attention scores for given query
    print(attn_score_2)
    return (attn_score_2,)


@app.cell
def _(attn_score_2, keys, torch):
    d_k = keys.shape[1]
    attn_wts_2 = torch.softmax(attn_score_2 / d_k**0.5, dim=-1)
    print(attn_wts_2)
    return (attn_wts_2,)


@app.cell
def _(attn_wts_2, values):
    ctx_vec_2 = attn_wts_2 @ values
    print(ctx_vec_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""impl SelfAttn """)
    return


@app.cell
def _(d_in, d_out, inputs, nn, torch):
    class SelfAttnV1(nn.Module):

        def __init__(self, d_in, d_out):
            super().__init__()
            self.W_query = nn.Parameter(torch.rand(d_in, d_out))
            self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
            self.W_value = nn.Parameter(torch.rand(d_in, d_out))

        def forward(self, x):
            keys = x @ self.W_key
            queries = x @ self.W_query
            values = x @ self.W_value

            attn_scores = queries @ keys.T # omega
            attn_weights = torch.softmax(
                attn_scores / keys.shape[-1]**0.5, dim=-1
            )

            context_vec = attn_weights @ values
            return context_vec

    torch.manual_seed(123)
    _sa_v1 = SelfAttnV1(d_in, d_out)
    print(_sa_v1(inputs))
    return


@app.cell
def _(d_in, d_out, inputs, nn, torch):
    class SelfAttnV2(nn.Module):

        def __init__(self, d_in, d_out, qkv_bias=False):
            super().__init__()
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        def forward(self, x):
            keys = self.W_key(x)
            queries = self.W_query(x)
            values = self.W_value(x)

            attn_scores = queries @ keys.T
            attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

            context_vec = attn_weights @ values
            return context_vec

    torch.manual_seed(789)
    sa_v2 = SelfAttnV2(d_in, d_out)
    print(sa_v2(inputs))
    return (sa_v2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""hiding future words w causal attn""")
    return


@app.cell
def _(attn_scores, inputs, keys, sa_v2, torch):
    sa_v2_queries = sa_v2.W_query(inputs)
    sa_v2_keys = sa_v2.W_key(inputs) 
    sa_v2_attn_scores = sa_v2_queries @ sa_v2_keys.T

    sa_v2_attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    print(sa_v2_attn_weights)
    return


@app.cell
def _(attn_scores, torch):
    context_length = attn_scores.shape[0]
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    print(mask_simple)
    return context_length, mask_simple


@app.cell
def _(attn_weights, mask_simple):
    masked_simple = attn_weights*mask_simple
    print(masked_simple)
    return (masked_simple,)


@app.cell
def _(masked_simple):
    row_sums = masked_simple.sum(dim=-1, keepdim=True)
    masked_simple_norm = masked_simple / row_sums
    print(masked_simple_norm)
    return


@app.cell
def _(attn_scores, context_length, torch):
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
    print(masked)
    return (masked,)


@app.cell
def _(keys, masked, torch):
    _attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
    print(_attn_weights)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""impl compact causal self-attn class""")
    return


@app.cell
def _(inputs, torch):
    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape) # 2 inputs with 6 tokens each, and each token has embedding dimension 3
    return (batch,)


@app.cell
def _(batch, context_length, d_in, d_out, nn, torch):
    class CausalAttention(nn.Module):

        def __init__(self, d_in, d_out, context_length,
                     dropout, qkv_bias=False):
            super().__init__()
            self.d_out = d_out
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.dropout = nn.Dropout(dropout) # New
            self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

        def forward(self, x):
            b, num_tokens, d_in = x.shape # New batch dimension b
            # For inputs where `num_tokens` exceeds `context_length`, this will result in errors
            # in the mask creation further below.
            # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs  
            # do not exceed `context_length` before reaching this forward method. 
            keys = self.W_key(x)
            queries = self.W_query(x)
            values = self.W_value(x)

            attn_scores = queries @ keys.transpose(1, 2) # Changed transpose
            attn_scores.masked_fill_(  # New, _ ops are in-place
                self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
            attn_weights = torch.softmax(
                attn_scores / keys.shape[-1]**0.5, dim=-1
            )
            attn_weights = self.dropout(attn_weights) # New

            context_vec = attn_weights @ values
            return context_vec

    torch.manual_seed(123)

    ca_context_length = batch.shape[1]
    ca = CausalAttention(d_in, d_out, context_length, 0.0)

    context_vecs = ca(batch)

    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)
    return (CausalAttention,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""multi head attn wrapper""")
    return


@app.cell
def _(CausalAttention, batch, nn, torch):
    class MultiHeadAttentionWrapper(nn.Module):

        def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
            super().__init__()
            self.heads = nn.ModuleList(
                [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
                 for _ in range(num_heads)]
            )

        def forward(self, x):
            return torch.cat([head(x) for head in self.heads], dim=-1)


    torch.manual_seed(123)

    mha_context_length = batch.shape[1] # This is the number of tokens
    mha_d_in, mha_d_out = 3, 2
    mha = MultiHeadAttentionWrapper(
        mha_d_in, mha_d_out, mha_context_length, 0.0, num_heads=2
    )

    mha_context_vecs = mha(batch)

    print(mha_context_vecs)
    print("mha_context_vecs.shape:", mha_context_vecs.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Multi Head Attn""")
    return


@app.cell
def _(nn, torch):
    class MultiHeadAttention(nn.Module):
        def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
            super().__init__()
            assert (d_out % num_heads == 0), \
                "d_out must be divisible by num_heads"

            self.d_out = d_out
            self.num_heads = num_heads
            self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
            self.dropout = nn.Dropout(dropout)
            self.register_buffer(
                "mask",
                torch.triu(torch.ones(context_length, context_length),
                           diagonal=1)
            )

        def forward(self, x):
            b, num_tokens, d_in = x.shape
            # As in `CausalAttention`, for inputs where `num_tokens` exceeds `context_length`, 
            # this will result in errors in the mask creation further below. 
            # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs  
            # do not exceed `context_length` before reaching this forward method.

            keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
            queries = self.W_query(x)
            values = self.W_value(x)

            # We implicitly split the matrix by adding a `num_heads` dimension
            # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
            keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
            values = values.view(b, num_tokens, self.num_heads, self.head_dim)
            queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

            # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
            keys = keys.transpose(1, 2)
            queries = queries.transpose(1, 2)
            values = values.transpose(1, 2)

            # Compute scaled dot-product attention (aka self-attention) with a causal mask
            attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

            # Original mask truncated to the number of tokens and converted to boolean
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

            # Use the mask to fill attention scores
            attn_scores.masked_fill_(mask_bool, -torch.inf)

            attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Shape: (b, num_tokens, num_heads, head_dim)
            context_vec = (attn_weights @ values).transpose(1, 2) 

            # Combine heads, where self.d_out = self.num_heads * self.head_dim
            context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
            context_vec = self.out_proj(context_vec) # optional projection

            return context_vec
    return


if __name__ == "__main__":
    app.run()
