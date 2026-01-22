import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import random

from GPT2.GPT2 import GPT2
from GPT2.train_shakespear import train
from GPT2.dataset_add_numbers import AddNumData

def collate(batch):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)

    device = xs[0].device

    X = torch.zeros(len(xs), max_len, dtype=torch.long, device=device)
    Y = torch.full((len(xs), max_len), -1, dtype=torch.long, device=device)

    for i, (x, y) in enumerate(zip(xs, ys)):
        X[i, :x.size(0)] = x
        Y[i, :y.size(0)] = y

    return X, Y

@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig):

    # Model initialization
    # vocab_size = len(range(10)) + 2 # + and =
    max_num = 1000
    split_ab = True # for digits based splitting
    vocab_size = len(range(max_num)) + 2 if not split_ab else len(range(10)) + 2
    try:
        pad = cfg.pad
    except:
        pad = False
    try:
        model_path = cfg.model_path
    except:
        model_path = None
    model_path = model_path or "GPT2/Cache/gpt2_algo.pth"
    try:
        train_model = cfg.train_model
    except:
        train_model = False
    gpt2_algo = GPT2(
        vocab_size=vocab_size
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    if train_model:
        # DataLoader
        batch_size = 64
        train_dataset = AddNumData(max_num=max_num, pad=pad, device=next(gpt2_algo.parameters()).device, split_ab=split_ab)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate)
        val_dataset = AddNumData(max_num=max_num, pad=pad,device=next(gpt2_algo.parameters()).device, split_ab=split_ab)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate)

        # model training and saving
        optimizer = train(
            max_iters=5000,
            model=gpt2_algo,
            train_dataloader=train_dataloader, 
            val_dataloader=val_dataloader,
        )

        checkpoints = {
            "model_state_dict": gpt2_algo.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }

        try:
            torch.save(checkpoints, model_path)
            print("Algo Model saved successfully")
        except Exception as e:
            raise e
    else:
        checkpoint = torch.load(model_path, map_location=next(gpt2_algo.parameters()).device)
        gpt2_algo.load_state_dict(checkpoint["model_state_dict"])
        print("Model loaded from disk")

    # generating
    max_size = len(str(2 * (max_num-1)))
    andd = AddNumData(max_num=max_num, pad=True, split_ab=True)
    a = random.randint(a=0, b=max_num) #[a,b]
    b = random.randint(a=0, b=max_num) #[a,b]
    c = a + b
    # we are ignoring c here
    context: list[int] = andd.encode(list(AddNumData.pad(a, b, c, max_size, True, pad).split("=")[0] + "="))
    print("context:", context)
    context: torch.Tensor = torch.tensor(context).reshape(1,-1).to(next(gpt2_algo.parameters()).device)
    generated_tokens = gpt2_algo.generate(idx=context, max_new_tokens=4)
    # generated_tokens_str = [str(tk) for tk in generated_tokens[0].tolist()]
    # generated_text = "".join(generated_tokens_str)
    print(andd.decode(generated_tokens.tolist()[0]))


if __name__ == "__main__":
    main()