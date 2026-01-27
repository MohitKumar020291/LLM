import os, math, torch, hydra, pickle as pkl
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from utils import read_web_pages, get_corpus
from GPT2.GPT2 import GPT2, Data
from Tokenizer.BPE import Tokenizer
from tokenizers import Tokenizer as HFTokenizer
from transformers import AutoTokenizer


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.barrier()
        return local_rank
    return 0


def lr_lambda(it, warmup_iters, max_iters):
    if it < warmup_iters:
        return it / warmup_iters
    progress = (it - warmup_iters) / (max_iters - warmup_iters)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters):
    out = {}
    model.eval()
    for loader, split in [(train_loader, "train"), (val_loader, "val")]:
        losses = torch.zeros(eval_iters)
        it = iter(loader)
        for k in range(eval_iters):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses
    model.train()
    return out


def train(model, optimizer, train_loader, val_loader, train_sampler,
          num_epochs=10, warmup_iters=500, max_steps=50_000,
          eval_interval=500, eval_iters=200, ckpt_path="GPT2/Cache/interrupt_ckpt.pth"):

    if not os.path.exists(Path(ckpt_path).parent):
        print(f"Parent path of ckpt_path = {ckpt_path} does not exist!")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda it: lr_lambda(it, warmup_iters, max_steps)
    )

    # scaler = torch.cuda.amp.GradScaler()
    step = 0

    try:
        for epoch in range(num_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            for x, y in train_loader:
                if step >= max_steps:
                    return optimizer

                if step % eval_interval == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
                    losses = estimate_loss(model, train_loader, val_loader, eval_iters)
                    print(f"step {step}: train {losses['train'].mean():.4f}, val {losses['val'].mean():.4f}")

                x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
                with torch.cuda.amp.autocast():
                    _, loss = model(x, y)

                optimizer.zero_grad()
                loss.backward()
                # scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                # scaler.step(optimizer)
                # scaler.update()
                scheduler.step()

                step += 1

    except KeyboardInterrupt:
        if not dist.is_initialized() or dist.get_rank() == 0:
            print("\n[CTRL+C] Saving checkpoint at step", step)
            torch.save({
                "step": step,
                "model_state_dict": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, ckpt_path)
        raise

    return optimizer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    training_corpus_urls = OmegaConf.select(cfg, "training_corpus_urls", default=None)
    training_corpus_path = OmegaConf.select(cfg, "training_corpus_path", default=None)
    train_model = OmegaConf.select(cfg, "train_model", default=False)
    tokenizer_path_pkl = OmegaConf.select(cfg, "tokenizer_path_pkl", default="Tokenizer/Cache/Tokenizers/tokenizer.pkl")
    tokenizer_path_hf = OmegaConf.select(cfg, "tokenizer_path_hf", default="Tokenizer/Cache/Tokenizers/HFTokenizer")
    tokenizer_type = OmegaConf.select(cfg, "tokenizer_type", default="hf")
    print(tokenizer_type)
    tokenizer_path = tokenizer_path_hf if tokenizer_type == "hf" else tokenizer_path_pkl
    model_path = OmegaConf.select(cfg, "model_path", default="GPT2/Cache/gpt_s.pth")
    generate = OmegaConf.select(cfg, "generate", default=False)
    continue_training = OmegaConf.select(cfg, "continue_training", default=False)
    block_size = OmegaConf.select(cfg, "block_size", default=256)
    lr=3e-4

    if training_corpus_path and not os.path.exists(training_corpus_path):
        raise ValueError(f"Path {training_corpus_path} does not exist!")
    if not(os.path.exists(Path(model_path).parent)):
        raise ValueError(f"{model_path}'s parent directory = {Path(model_path).parent} does not exist!")

    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    print("tokenizer path", tokenizer_path)
    if os.path.isdir(tokenizer_path):
        # HF tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print("Using HF tokenizer!")
    else:
        with open(tokenizer_path, 'rb') as f:
            tokenizer: Tokenizer = pkl.load(f)

    if isinstance(tokenizer, HFTokenizer):
        vocab_size = tokenizer.get_vocab_size()
    else:
        vocab_size = len(tokenizer) # so this is more than the actual tokens, 8_000
        # vocab_size = tokenizer.vocab_size
    vocab_size = len(tokenizer.vocab) if not isinstance(tokenizer, HFTokenizer) else tokenizer.get_vocab_size()
    model = GPT2(vocab_size=vocab_size, block_size=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))
    if continue_training:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])

    corpus = get_corpus(training_corpus_path)
    pages = read_web_pages(training_corpus_urls) if train_model else []
    corpus += "\n".join(pages)

    train_dataset = Data(corpus=corpus, tokenizer=tokenizer, train_size=0.9, split="train", block_size=256, device=device)
    val_dataset   = Data(corpus=corpus, tokenizer=tokenizer, train_size=0.9, split="val", block_size=256, device=device)

    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    val_sampler   = DistributedSampler(val_dataset, shuffle=False) if dist.is_initialized() else None

    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler, shuffle=(train_sampler is None))
    val_loader   = DataLoader(val_dataset,   batch_size=64, sampler=val_sampler,   shuffle=False)

    try:
        optimizer = train(model, optimizer, train_loader, val_loader, train_sampler) if train_model else None
    except Exception as e:
        raise e

    if train_model and (not dist.is_initialized() or dist.get_rank() == 0):
        torch.save({
            "model_state_dict": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, model_path)

    if generate:
        ...


if __name__ == "__main__":
    main()