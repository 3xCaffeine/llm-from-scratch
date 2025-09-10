import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    return (torch,)


@app.cell
def _(torch):
    torch.__version__
    return


@app.cell
def _(torch):
    inputs = torch.tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55],
    ])
    return (inputs,)


@app.cell
def _(inputs):
    input_query = inputs[1]
    input_query
    return


if __name__ == "__main__":
    app.run()
