import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import tiktoken
    import os
    import time
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    
    # Setup project imports
    from utils import setup_project_imports
    setup_project_imports()
    
    from gpt_model import GPTModel, create_dataloader_v1
    
    return (
        GPTModel,
        MaxNLocator,
        create_dataloader_v1,
        mo,
        os,
        plt,
        tiktoken,
        time,
        torch,
    )


@app.cell
def _(GPTModel, torch):
    GPT_CONFIG_124M = {
        "vocab_size": 50257,   # Vocabulary size
        "context_length": 256, # Shortened context length
        "emb_dim": 768,        # Embedding dimension
        "n_heads": 12,         # Number of attention heads
        "n_layers": 12,        # Number of layers
        "drop_rate": 0.1,      # Dropout rate
        "qkv_bias": False      # Query-key-value bias
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval();  # Disable dropout during inference
    return GPT_CONFIG_124M, model


@app.cell
def _(torch):
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
    return (generate_text_simple,)


@app.cell
def _(GPT_CONFIG_124M, generate_text_simple, model, tiktoken, torch):
    def text_to_token_ids(text, tokenizer):
        encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
        return encoded_tensor

    def token_ids_to_text(token_ids, tokenizer):
        flat = token_ids.squeeze(0) # remove batch dimension
        return tokenizer.decode(flat.tolist())

    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    return text_to_token_ids, token_ids, token_ids_to_text, tokenizer


@app.cell
def _(torch):
    inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                           [40,    1107, 588]])   #  "I really like"]

    targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                            [1107,  588, 11311]]) #  " really like chocolate"]
    return inputs, targets


@app.cell
def _(inputs, model, torch):
    with torch.no_grad():
        logits = model(inputs)

    probas = torch.softmax(logits, dim=-1) # Probability of each token in vocabulary
    print(probas.shape) # Shape: (batch_size, num_tokens, vocab_size)
    return logits, probas


@app.cell
def _(probas, targets, token_ids, token_ids_to_text, tokenizer, torch):
    _token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print("Token IDs:\n", _token_ids)

    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
    return


@app.cell
def _(probas, targets):
    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 1:", target_probas_1)

    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 2:", target_probas_2)
    return target_probas_1, target_probas_2


@app.cell
def _(target_probas_1, target_probas_2, torch):
    # Compute logarithm of all token probabilities
    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print(log_probas)
    return (log_probas,)


@app.cell
def _(log_probas, torch):
    # Calculate the average probability for each token
    avg_log_probas = torch.mean(log_probas)
    print(avg_log_probas)
    return (avg_log_probas,)


@app.cell
def _(avg_log_probas):
    neg_avg_log_probas = avg_log_probas * -1
    print(neg_avg_log_probas)
    return


@app.cell
def _(logits, targets):
    # Logits have shape (batch_size, num_tokens, vocab_size)
    print("Logits shape:", logits.shape)

    # Targets have shape (batch_size, num_tokens)
    print("Targets shape:", targets.shape)
    return


@app.cell
def _(logits, targets):
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()

    print("Flattened logits:", logits_flat.shape)
    print("Flattened targets:", targets_flat.shape)
    return logits_flat, targets_flat


@app.cell
def _(logits_flat, targets_flat, torch):
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print(loss)
    return (loss,)


@app.cell
def _(loss, torch):
    perplexity = torch.exp(loss)
    print(perplexity)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""calculating the training and validation set losses""")
    return


@app.cell
def _(os):
    file_path = os.path.join(os.path.dirname(__file__), "../working_with_text_data/the-verdict.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    return (text_data,)


@app.cell
def _(text_data):
    print(text_data[:99])
    print(text_data[-99:])
    return


@app.cell
def _(text_data, tokenizer):
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))

    print("Characters:", total_characters)
    print("Tokens:", total_tokens)
    return (total_tokens,)


@app.cell
def _(GPT_CONFIG_124M, create_dataloader_v1, text_data, torch):
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]


    torch.manual_seed(123)

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    return train_loader, train_ratio, val_loader


@app.cell
def _(GPT_CONFIG_124M, total_tokens, train_ratio):
    # Sanity check

    if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Not enough tokens for the training loader. "
              "Try to lower the `GPT_CONFIG_124M['context_length']` or "
              "increase the `training_ratio`")

    if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Not enough tokens for the validation loader. "
              "Try to lower the `GPT_CONFIG_124M['context_length']` or "
              "decrease the `training_ratio`")
    return


@app.cell
def _(train_loader, val_loader):
    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)

    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)
    return


@app.cell
def _(train_loader, val_loader):
    train_tokens = 0
    for input_batch, target_batch in train_loader:
        train_tokens += input_batch.numel()

    val_tokens = 0
    for input_batch, target_batch in val_loader:
        val_tokens += input_batch.numel()

    print("Training tokens:", train_tokens)
    print("Validation tokens:", val_tokens)
    print("All tokens:", train_tokens + val_tokens)
    return


@app.cell
def _(torch):
    def calc_loss_batch(input_batch, target_batch, model, device):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss


    def calc_loss_loader(data_loader, model, device, num_batches=None):
        total_loss = 0.
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            # Reduce the number of batches to match the total number of batches in the data loader
            # if num_batches exceeds the number of batches in the data loader
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches
    return calc_loss_batch, calc_loss_loader


@app.cell
def _(calc_loss_loader, model, torch, train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes


    torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader

    with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Train LLM""")
    return


@app.cell
def _(
    calc_loss_batch,
    calc_loss_loader,
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
    torch,
):
    def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                           eval_freq, eval_iter, start_context, tokenizer):
        # Initialize lists to track losses and tokens seen
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1

        # Main training loop
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode

            for input_batch, target_batch in train_loader:
                optimizer.zero_grad() # Reset loss gradients from previous batch iteration
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward() # Calculate loss gradients
                optimizer.step() # Update model weights using loss gradients
                tokens_seen += input_batch.numel()
                global_step += 1

                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                          f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            # Print a sample text after each epoch
            generate_and_print_sample(
                model, tokenizer, device, start_context
            )

        return train_losses, val_losses, track_tokens_seen


    def evaluate_model(model, train_loader, val_loader, device, eval_iter):
        model.eval()
        with torch.no_grad():
            train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
            val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()
        return train_loss, val_loss


    def generate_and_print_sample(model, tokenizer, device, start_context):
        model.eval()
        context_size = model.pos_emb.weight.shape[0]
        encoded = text_to_token_ids(start_context, tokenizer).to(device)
        with torch.no_grad():
            token_ids = generate_text_simple(
                model=model, idx=encoded,
                max_new_tokens=50, context_size=context_size
            )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
        model.train()
    return (train_model_simple,)


@app.cell
def _(
    GPTModel,
    GPT_CONFIG_124M,
    MaxNLocator,
    device,
    plt,
    time,
    tokenizer,
    torch,
    train_loader,
    train_model_simple,
    val_loader,
):
    def _():
        start_time = time.time()

        torch.manual_seed(123)
        model = GPTModel(GPT_CONFIG_124M)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

        num_epochs = 10
        train_losses, val_losses, tokens_seen = train_model_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs, eval_freq=5, eval_iter=5,
            start_context="Every effort moves you", tokenizer=tokenizer
        )

        # Plot the losses after training
        epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        print(f"Training completed in {execution_time_minutes:.2f} minutes.")

        # Return the trained model for further use
        return model


    def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
        fig, ax1 = plt.subplots(figsize=(5, 3))

        # Plot training and validation loss against epochs
        ax1.plot(epochs_seen, train_losses, label="Training loss")
        ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

        # Create a second x-axis for tokens seen
        ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
        ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
        ax2.set_xlabel("Tokens seen")

        fig.tight_layout()  # Adjust layout to make room
        # plt.savefig("loss-plot.pdf")
        plt.show()

    trained_model = _()
    return (trained_model,)


@app.cell
def _(
    GPT_CONFIG_124M,
    generate_text_simple,
    text_to_token_ids,
    tiktoken,
    token_ids_to_text,
    torch,
    trained_model,
):
    inference_device = torch.device("cpu")

    trained_model.to(inference_device)
    trained_model.eval()

    inference_tokenizer = tiktoken.get_encoding("gpt2")

    inference_token_ids = generate_text_simple(
        model=trained_model,
        idx=text_to_token_ids("Every effort moves you", inference_tokenizer).to(inference_device),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    print("Output text:\n", token_ids_to_text(inference_token_ids, inference_tokenizer))
    return


if __name__ == "__main__":
    app.run()
