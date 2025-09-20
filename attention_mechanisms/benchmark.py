import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    from multi_head_attn_impl import MultiHeadAttentionCombinedQKV, MHAPyTorchFlexAttention, MHAPyTorchScaledDotProduct
    import timeit
    import matplotlib.pyplot as plt
    import numpy as np
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    return (
        MHAPyTorchFlexAttention,
        MHAPyTorchScaledDotProduct,
        MultiHeadAttentionCombinedQKV,
        mo,
        np,
        plt,
        timeit,
        torch,
    )


@app.cell
def _(torch):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Running on {device}")
    return (device,)


@app.cell
def _(device, torch):
    batch_size = 8
    context_len = 1024
    embed_dim = 768
    embeddings = torch.randn((batch_size, context_len, embed_dim), device=device)
    print(embeddings)
    return (embeddings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""speed comparison""")
    return


@app.cell
def _(MultiHeadAttentionCombinedQKV, device, embeddings, timeit):
    mha_combined_qkv = MultiHeadAttentionCombinedQKV(768, 768, 12, 1024).to(device)
    _times = timeit.repeat(lambda: mha_combined_qkv(embeddings), repeat=7, number=1)
    _mean_time = sum(_times) / len(_times)
    _std_time = (sum((t - _mean_time)**2 for t in _times) / len(_times))**0.5
    print(f"{_mean_time*1000:.1f} ms ± {_std_time*1000:.0f} µs per loop (mean ± std. dev. of 7 runs, 1 loop each)")
    return (mha_combined_qkv,)


@app.cell
def _(MHAPyTorchScaledDotProduct, device, embeddings, timeit):
    mha_pytorch_sdp = MHAPyTorchScaledDotProduct(768, 768, 12, 1024).to(device)
    _times = timeit.repeat(lambda: mha_pytorch_sdp(embeddings), repeat=7, number=1)
    _mean_time = sum(_times) / len(_times)
    _std_time = (sum((t - _mean_time)**2 for t in _times) / len(_times))**0.5
    print(f"{_mean_time*1000:.1f} ms ± {_std_time*1000:.0f} µs per loop (mean ± std. dev. of 7 runs, 1 loop each)")
    return (mha_pytorch_sdp,)


@app.cell
def _(MHAPyTorchFlexAttention, device, embeddings, timeit):
    mha_pytorch_flex = MHAPyTorchFlexAttention(768, 768, 12, 1024).to(device)
    _times = timeit.repeat(lambda: mha_pytorch_flex(embeddings), repeat=7, number=1)
    _mean_time = sum(_times) / len(_times)
    _std_time = (sum((t - _mean_time)**2 for t in _times) / len(_times))**0.5
    print(f"{_mean_time*1000:.1f} ms ± {_std_time*1000:.0f} µs per loop (mean ± std. dev. of 7 runs, 1 loop each)")
    return (mha_pytorch_flex,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""viz""")
    return


@app.cell
def _(mha_combined_qkv, mha_pytorch_flex, mha_pytorch_sdp):
    functions = {
        "1) MHA with combined QKV weights": mha_combined_qkv,
        "2) MHA with PyTorch scaled_dot_product_attention": mha_pytorch_sdp,
        "3) PyTorch's FlexAttention":  mha_pytorch_flex
    }
    return (functions,)


@app.cell
def _(plt):
    plt.rcParams["figure.facecolor"] = "#121212"
    plt.rcParams["axes.facecolor"] = "#121212"
    plt.rcParams["axes.edgecolor"] = "white"
    plt.rcParams["axes.labelcolor"] = "white"
    plt.rcParams["text.color"] = "white"
    plt.rcParams["xtick.color"] = "white"
    plt.rcParams["ytick.color"] = "white"
    plt.rcParams["grid.color"] = "#444444"
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["lines.markersize"] = 8

    def plot_execution_times(functions, __execution_means, __execution_stds, filename):

        # Create plot
        fig, ax = plt.subplots()
        bars = ax.bar(functions.keys(), __execution_means, yerr=__execution_stds, capsize=5, error_kw={'ecolor': 'grey'})

        plt.ylabel("Execution time (ms)")
        plt.xticks(rotation=45, ha="right")

        # Calculate new ylim with a margin
        max_execution_time = max(__execution_means)
        upper_ylim = max_execution_time + 0.4 * max_execution_time  # Adding a 40% margin
        plt.ylim(0, upper_ylim)

        # Annotate bars with execution times
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + (0.05 * upper_ylim), round(yval, 2), ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
    return (plot_execution_times,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""speed comparison w warmup (forward pass only)""")
    return


@app.cell
def _(np, torch):
    def time_pytorch_function(func, *input, num_repeats=1_000):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Warmup
        for _ in range(5):
            func(*input)
        torch.cuda.synchronize()

        times = []
        for _ in range(num_repeats):
            start.record()
            func(*input)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        return np.mean(times), np.std(times)
    return (time_pytorch_function,)


@app.cell
def _(embeddings, functions, plot_execution_times, time_pytorch_function):
    _execution_stats = [time_pytorch_function(fn, embeddings) for fn in functions.values()]
    _execution_means = [stat[0] for stat in _execution_stats]
    _execution_stds = [stat[1] for stat in _execution_stats]
    plot_execution_times(functions, _execution_means, _execution_stds)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""speed comparison w warmup (forward and backward pass)""")
    return


@app.cell
def _(np, torch):
    def forward_backward(func, embeddings):
        if embeddings.grad is not None:
            embeddings.grad.zero_()

        output = func(embeddings)
        loss = output.sum()
        loss.backward()


    def time_pytorch_function_forward_backward(func, *input, num_repeats = 1_000):
        # CUDA IS ASYNC so can't use python time module
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Warmup
        for _ in range(5):
            forward_backward(func, *input)
        torch.cuda.synchronize()

        times = []
        for _ in range(num_repeats):
            start.record()
            forward_backward(func, *input)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        return np.mean(times), np.std(times)
    return (time_pytorch_function_forward_backward,)


@app.cell
def _(
    embeddings,
    functions,
    plot_execution_times,
    time_pytorch_function_forward_backward,
):
    _execution_stats = [time_pytorch_function_forward_backward(fn, embeddings) for fn in functions.values()]
    _execution_means = [stat[0] for stat in _execution_stats]
    _execution_stds = [stat[1] for stat in _execution_stats]


    plot_execution_times(functions, _execution_means, _execution_stds)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""speed comparison w warmup and compilation (forward and backward pass)""")
    return


@app.cell
def _(torch):
    def prepare_function(fn):
        fn = torch.compile(fn)
        return fn
    return (prepare_function,)


@app.cell
def _(
    embeddings,
    functions,
    plot_execution_times,
    prepare_function,
    time_pytorch_function_forward_backward,
):
    _execution_stats = [time_pytorch_function_forward_backward(prepare_function(fn), embeddings) for fn in functions.values()]
    _execution_means = [stat[0] for stat in _execution_stats]
    _execution_stds = [stat[1] for stat in _execution_stats]


    plot_execution_times(functions, _execution_means, _execution_stds)
    return


if __name__ == "__main__":
    app.run()
