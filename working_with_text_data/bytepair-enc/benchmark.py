import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import tiktoken
    import os
    import timeit
    import statistics
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from transformers import AutoTokenizer, GPT2TokenizerFast
    from minbpe import GPT4Tokenizer
    return (
        GPT2TokenizerFast,
        GPT4Tokenizer,
        mo,
        os,
        statistics,
        tiktoken,
        timeit,
    )


@app.cell
def _(statistics):
    def format_timeit(_times, loops):
        times_per_loop = [t / loops for t in _times]
        mean = statistics.mean(times_per_loop)
        std = statistics.stdev(times_per_loop)
        def format_time(t):
            if t >= 1:
                return f"{t:.2f} s"
            elif t >= 0.001:
                return f"{t*1000:.2f} ms"
            else:
                return f"{t*1000000:.1f} μs"
        mean_str = format_time(mean)
        std_str = format_time(std)
        return f"{mean_str} ± {std_str} per loop (mean ± std. dev. of {len(_times)} runs, {loops} loops each)"
    return (format_timeit,)


@app.cell
def _(os):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    verdict_path = os.path.join(script_dir, "..", "the-verdict.txt")
    with open(verdict_path, "r", encoding="utf-8") as f:
        text = f.read()
    return (text,)


@app.cell
def _(mo):
    mo.md(r"""tiktoken tokenizer""")
    return


@app.cell
def _(format_timeit, text, tiktoken, timeit):
    enc = tiktoken.get_encoding("cl100k_base")
    _times = timeit.repeat(lambda: enc.encode(text), repeat=7, number=100)
    print(format_timeit(_times, 100))
    return


@app.cell
def _(mo):
    mo.md(r"""HF GPT4 tokenizer """)
    return


@app.cell
def _(GPT2TokenizerFast, format_timeit, text, timeit):
    tokenizer = GPT2TokenizerFast.from_pretrained("Xenova/gpt-4")
    _times = timeit.repeat(lambda: tokenizer.encode(text), repeat=7, number=100)
    print(format_timeit(_times, 100))
    return


@app.cell
def _(mo):
    mo.md(r"""custom minbpe tokenizer""")
    return


@app.cell
def _(GPT4Tokenizer, format_timeit, text, timeit):
    custom_tokenizer = GPT4Tokenizer()
    _times = timeit.repeat(lambda: custom_tokenizer.encode(text), repeat=7, number=100)
    print(format_timeit(_times, 100))
    return


if __name__ == "__main__":
    app.run()
