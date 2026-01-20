import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
import pickle as pkl

from utils import read_web_pages, get_corpus
from GPT2.GPT2 import GPT2, Data
from Tokenizer.BPE import Tokenizer
import os
import math


def lr_lambda(it, warmup_iters, max_iters):
    if it < warmup_iters:
        return it / warmup_iters
    progress = (it - warmup_iters) / (max_iters - warmup_iters)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

@torch.no_grad()
def estimate_loss(
    model: GPT2,
    train_dataloader,
    val_dataloader, 
    eval_iters: int, 
):
    out = {}
    model.eval() #crucial
    for dataloader, split in zip([train_dataloader, val_dataloader], ["train", "val"]):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = next(iter(dataloader))
            device = next(model.parameters()).device
            X = X.to(device)
            Y = Y.to(device)
            # shift to GPU
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses
    model.train()
    return out

def train(
    model: GPT2,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    warmup_iters: int = 500,
    max_iters: int = 2000,
    eval_interval: int = 500, 
    eval_iters: int = 200,
    learning_rate: float = 3e-4
):
    """
        returns: optimizer used and whether to save the model details or not?
    """
    print("Starting training...")
    loss = 0
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lambda it: lr_lambda(it, warmup_iters, max_iters)
                )
    for iter_ in range(max_iters):
        if iter_ % eval_interval == 0:
            losses = estimate_loss(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, eval_iters=eval_iters)
            print("step %d: train loss %.4f, val loss %.4f" % (iter, losses['train'].mean(), losses['val'].mean()))

        try:
            x_batches, y_batches = next(iter(train_dataloader)) #(B,T)
        except Exception as e:
            raise e
        logits, loss = model(x_batches, y_batches)
        optimizer.zero_grad()
        loss.backward()
        print(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    return optimizer

@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig):
    try:
        train_model = cfg.train_model
    except:
        train_model = False
    try:
        continue_training = cfg.continue_training
    except:
        continue_training = False
    try:
        estimate_loss_only = cfg.estimate_loss_only
    except:
        estimate_loss_only = False
    try:
        training_corpus_urls = cfg.training_corpus_urls
    except:
        training_corpus_urls = None
    
    try:
        training_corpus_path = cfg.training_corpus_path
    except:
        training_corpus_path = None

    default_tokenizer_path = "Tokenizer/Cache/Tokenizers/tokenizer.pkl"
    try:
        generate = cfg.generate
    except:
        generate = False

    try:
        tokenizer_path = cfg.tokenizer_path
    except:
        print("Using default tokenizer path")
        tokenizer_path = "Tokenizer/Cache/Tokenizers/tokenizer.pkl"

    if os.path.exists(tokenizer_path):
        print("Tokenizer path exists:", tokenizer_path)
    else:
        print(f"Tokenizer path: {tokenizer_path} does not exist using default: {default_tokenizer_path}")
        tokenizer_path = default_tokenizer_path

    # I am just picking up a tokenizer irrespective of the corpus they have been trained on
    with open(tokenizer_path, 'rb') as file:
        tokenizer: Tokenizer = pkl.load(file)

    # Initialize model
    vocab_size = len(tokenizer.vocab) #currently the size if 1000
    model_path = "GPT2/Cache/gpt2.pth"
    corpus = get_corpus(corpus_path=training_corpus_path)
    pages = read_web_pages(urls=training_corpus_urls) if train_model else []
    pages = "\n".join(pages)
    corpus += "\n".join(pages.split("\n"))
    print("===========CORPUS============")
    print(corpus[:500])
    print("=============================")
    train_size = 0.9
    block_size = 256
    batch_size = 64
    model = GPT2(vocab_size=vocab_size, block_size=block_size).to("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = Data(
        corpus=corpus, 
        tokenizer=tokenizer, 
        train_size=train_size,
        split="train",
        block_size=block_size,
        device=next(model.parameters()).device
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = Data(
        corpus=corpus, 
        tokenizer=tokenizer, 
        train_size=train_size,
        split="val",
        block_size=block_size,
        device=next(model.parameters()).device
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print("Model initialized on device:", next(model.parameters()).device)

    if continue_training:
        checkpoint = torch.load(model_path, map_location=next(model.parameters()).device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Training the model
    if train_model and not estimate_loss_only:
        try:
            optimizer = train(
                            model=model, 
                            train_dataloader=train_dataloader, 
                            val_dataloader=val_dataloader
                        )
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, model_path)
        except Exception as e:
            print("An error occurred during training:", e)
    else:
        checkpoint = torch.load(model_path, map_location=next(model.parameters()).device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded from disk.")

    
    if estimate_loss_only:
        losses = estimate_loss(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, eval_iters=100)
        print("train loss %.4f, val loss %.4f" % (losses['train'].mean(), losses['val'].mean()))
        return

    # generating from the model
    if generate:
        context = torch.zeros((1, 1), dtype=torch.long, device=next(model.parameters()).device)
        generated_tokens = model.generate(idx=context, max_new_tokens=500)
        generated_text = tokenizer.decode(generated_tokens[0].tolist())
        # print new lines if generated
        print("===========GENERATED TEXT============")
        print(generated_text)

if __name__ == "__main__":
    main()