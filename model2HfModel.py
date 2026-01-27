import os
import hydra
from omegaconf import OmegaConf, DictConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from utils import save_as_hf_model
import hf_registry

@hydra.main(version_base=None, config_name=None, config_path=None)
def main(cfg: DictConfig):
    save = OmegaConf.select(cfg, "save", default=True)
    # model_path = OmegaConf.select(cfg, "model_path", default="GPT2/Cache/gpt_fs.pth")
    model_path = OmegaConf.select(cfg, "model_path", default="GPT2/Cache/interrupt_ckpt.pth")

    model_save_dir = "ModelSaveTestDir"

    if save:
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        save_as_hf_model(
            model_path=model_path, 
            model_save_dir=model_save_dir,
            tokenizer_path="Tokenizer/Cache/Tokenizers/HF_tokenizer"
        )


    # this is for testing
    AutoConfig.from_pretrained(model_save_dir)
    AutoTokenizer.from_pretrained(model_save_dir)
    AutoModelForCausalLM.from_pretrained(model_save_dir)

if __name__ == "__main__":
    main()