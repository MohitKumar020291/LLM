import hydra
from omegaconf import OmegaConf, DictConfig
from unsloth import FastLanguageModel
from datasets import load_dataset
import tempfile
from transformers import TrainingArguments, Trainer
from functools import partial

from utils import read_web_page
import hf_registry


def tokenize(tokenizer, example, max_seq_length: int = 256):
    out = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
    )
    out["labels"] = out["input_ids"].copy()
    return out


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    causal = OmegaConf.select(cfg, "causal", default=True)
    max_seq_length = OmegaConf.select(cfg, "block_size", default=256)
    finetune_corpus_url = OmegaConf.select(cfg, "finetune_corpus_url", default=None)

    print("Causal", causal)
    print("max_seq_length", max_seq_length)
    print("finetune_corpus_url", finetune_corpus_url)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="ModelSaveTestDir",
        max_seq_length=256,
        load_in_4bit=True,
        attn_implementation="eager"
    )
    model.resize_token_embeddings(len(tokenizer))

    # getting peft_model - a full model but the gradient does only flow for few!
    # Injecting loRA - goes on:
    # Attention projections (Q, K, V, O) :- fused attention's q, k, v, proj
    # Feed-forward linear layers :- net[0], net[2]
    # Optionally output head (lm_head / mlp)
    # One more thing the model we are using here is `GPT2Causal` but it still works because
    # hf peft works on suffix not prefix
    target_modules = [
        "query",
        "key",
        "value",
        "proj",
        "net.0",
        "net.2",
    ]

    model = FastLanguageModel.get_peft_model(
        model,
        r=16, #rank of loRA matrices
        lora_alpha=32, #scaling weights of adapter
        lora_dropout=0.05,
        bias="none", #trying different approaches
        target_modules=target_modules,
        use_gradient_checkpointing=True,
        random_state=42
    )

    #prepare dataset: tiny shakespear: this is causal (for chat/instruct, I might have to change it)
    # dataset = None
    # with tempfile.NamedTemporaryFile(mode='w+t', delete=True) as temp:
    #     temp_file_name = temp.name
    #     temp.write(read_web_page(finetune_corpus_url))
    #     temp.seek(0)
    #     dataset = load_dataset("text", data_files=temp_file_name)
    dataset = load_dataset("text", data_files={"train": finetune_corpus_url})

    tokenize_fn = partial(tokenize, tokenizer, max_seq_length=max_seq_length)
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # training with the trainer
    trainer = Trainer(
            model=model, #peft model
            train_dataset=tokenized["train"],
            args=TrainingArguments(
                per_device_train_batch_size=2, #I am using kaggle P100 single instance
                gradient_accumulation_steps=32, #processes 4 batches before updating grads
                learning_rate=2e-4,
                fp16=True,
                logging_steps=10,
                save_steps=500,
                output_dir="unsloth_out",
                num_train_epochs=1, #not more otherwise overfitting
                report_to="none",
        ),
    )
    try:
        trainer.train()
    except KeyboardInterrupt:
        pass

    # saving model
    model = model.merge_and_unload()
    model.save_pretrained("finetuned_model")
    tokenizer.save_pretrained("finetuned_model")


if __name__ == "__main__":
    main()