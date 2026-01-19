import torch
import hydra
from omegaconf import DictConfig
import pickle as pkl

from utils import read_web_page
from GPT2.GPT2 import GPT2
from Tokenizer.BPE import Tokenizer


@torch.no_grad()
def estimate_loss(model, eval_iters, block_size=32):
    out = {}
    model.eval() #crucial
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = model.get_batch(split, block_size=block_size)
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
    model, 
    max_iter: int = 5000, 
    batch_size: int = 64, 
    block_size: int = 256, 
    eval_interval: int = 500, 
    eval_iters: int = 200,
    learning_rate: float = 3e-4
):
    print("Starting training...")
    loss = 0
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    for iter in range(max_iter):
        if iter % eval_interval == 0:
            losses = estimate_loss(model=model, eval_iters=eval_iters)
            print("step %d: train loss %.4f, val loss %.4f" % (iter, losses['train'].mean(), losses['val'].mean()))
        x_batches, y_batches = model.get_batch(split="train", batch_size=batch_size, block_size=block_size) #(B,T)
        x_batches = x_batches.to(next(model.parameters()).device)
        y_batches = y_batches.to(next(model.parameters()).device)
        logits, loss = model(x_batches, y_batches)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return optimizer


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig):
    train_model = cfg.train_model
    generate = cfg.generate
    with open("Tokenizer/Cache/tokenizer.pkl", 'rb') as file:
        tokenizer: Tokenizer = pkl.load(file)

    vocab_size = len(tokenizer.vocab) #currently the size if 1000
    model_path = "GPT2/Cache/biagram_model.pth"
    corpus = ""
    pages = read_web_page(url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt") if train_model else ""
    corpus += "\n".join(pages.split("\n"))
    print("===========CORPUS============")
    print(corpus[:500])
    print("=============================")
    train_size = 0.9
    block_size = 32
    model = GPT2(vocab_size=vocab_size, corpus=corpus, tokenizer=tokenizer, 
                                train_size=train_size, block_size=block_size).to("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", next(model.parameters()).device)
    print("Model initialized.")
    if train_model:
        try:
            optimizer = train(model=model, block_size=block_size)
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

    # generating from the model
    if generate:
        context = torch.zeros((1, 1), dtype=torch.long, device=next(model.parameters()).device)
        generated_tokens = model.generate(idx=context, max_new_tokens=200)
        generated_text = tokenizer.decode(generated_tokens[0].tolist())
        # print new lines if generated
        print("===========GENERATED TEXT============")
        print(generated_text)

if __name__ == "__main__":
    main()