import matplotlib.pyplot as plt
import os
import torch
import tiktoken
import yaml

from gpt_model import GPTModel, create_dataloader_v1

def get_valid_file_path(prompt):
    while True:
        print(f"{prompt}: ")
        file_path = input()
        if os.path.exists(file_path):
            return file_path
        print("Error! File not found. Please enter a valid file path.")

def get_input(prompt, default, type_func=str):
    user_input = input(f"{prompt} (default: {default}): ")
    return type_func(user_input) if user_input else default

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


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
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
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

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

def main_training(gpt_config, settings, text_data):

    ##############################
    # Initialize model
    ##############################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(gpt_config)
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    ##############################
    # Set up dataloaders
    ##############################

    # Train/validation ratio
    train_ratio = settings["train_ratio"]
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    ##############################
    # Train model
    ##############################

    train_losses, val_losses, tokens_seen = train_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
    )

    return train_losses, val_losses, tokens_seen, model, optimizer

def main_generate(gpt_config, settings, model_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, weights_only=True)
    gpt = GPTModel(gpt_config)
    gpt.load_state_dict(checkpoint["model_state_dict"])
    gpt.to(device)
    gpt.eval()
    tokenizer = tiktoken.get_encoding("gpt2")

    while(True):

        print("\nEnter input prompt (Empty to exit):")
        input_prompt = input()
        if input_prompt == "":
            break

        token_ids = generate(
            model=gpt,
            idx=text_to_token_ids(input_prompt, tokenizer).to(device),
            max_new_tokens=settings["max_new_tokens"],
            context_size=gpt_config["context_length"],
            top_k=settings["top_k"],
            temperature=settings["temperature"]
        )

        print("Output text:", token_ids_to_text(token_ids, tokenizer))

if __name__ == "__main__":

    while(True):
        print("\n[[LLM Training and Generation App]]")
        print("What would you like to do?")
        print("1 - Training")
        print("2 - Generate")
        print("3 - Exit")
        chose = int(input())
        if (chose == 1):
            print("\nPlease enter the configurations for the model")

            # Get GPT configuration from user
            GPT_CONFIG_124M = {
                "vocab_size": get_input("Vocabulary size", 50257, int),
                "context_length": get_input("Context length", 1024, int),
                "emb_dim": get_input("Embedding dimension", 768, int),
                "n_heads": get_input("Number of attention heads", 12, int),
                "n_layers": get_input("Number of layers", 12, int),
                "drop_rate": get_input("Dropout rate", 0.1, float),
                "qkv_bias": get_input("Query-key-value bias (True/False)", False, lambda x: x.lower() == 'true')
            }

            # Get other settings from user
            TRAINING_SETTINGS = {
                "learning_rate": get_input("Learning rate", 5e-4, float),
                "num_epochs": get_input("Number of epochs", 10, int),
                "batch_size": get_input("Batch size", 2, int),
                "weight_decay": get_input("Weight decay", 0.1, float),
                "train_ratio": get_input("Train ratio", 0.9, float)
            }

            text_path = get_valid_file_path("Enter text file location")
            with open(text_path, "r", encoding="utf-8") as file:
                text_data = file.read()

            ###########################
            # Initiate training
            ###########################

            print("\nInitiating training....")
            train_losses, val_losses, tokens_seen, model, optimizer = main_training(GPT_CONFIG_124M, TRAINING_SETTINGS, text_data)

            ###########################
            # After training
            ###########################

            # Plot results
            epochs_tensor = torch.linspace(0, TRAINING_SETTINGS["num_epochs"], len(train_losses))
            plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
            plt.savefig("loss.pdf")

            # Save and load model

            print("\nTraining completed!")

            print("Enter the name of the model file to save:")
            model_name = input()
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, model_name + ".pth")

            print("Enter the name of the config file to save:")
            config_name = input()
            config = {
                "GPT_CONFIG_124M": GPT_CONFIG_124M,
                "OTHER_SETTINGS": TRAINING_SETTINGS
            }
            with open(config_name + ".yaml", 'w') as file:
                yaml.dump(config, file, default_flow_style=False)

            print(f"Model has saved to {model_name}.pth")
            print(f"Model has saved to {config_name}.yaml")

        elif (chose == 2):

            model_path = get_valid_file_path("\nEnter model file location")

            config_path = get_valid_file_path("Enter config file location")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            GPT_CONFIG_124M = config['GPT_CONFIG_124M']

            GENERATE_SETTINGS = {
                "max_new_tokens": get_input("Max tokens", 25, int),
                "top_k": get_input("Top k", 50, int),
                "temperature": get_input("Temperature", 1.0, float),
            }

            main_generate(GPT_CONFIG_124M, GENERATE_SETTINGS, model_path)

        else:
            break