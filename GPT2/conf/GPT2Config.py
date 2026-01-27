from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

class GPT2Config(PretrainedConfig):
    model_type = "my_gpt"

    def __init__(
        self,
        vocab_size: int = 8004,
        n_embed: int = 384,
        block_size: int = 256,
        num_heads: int = 4,
        n_layers: int = 6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.block_size = block_size
        self.num_heads = num_heads
        self.n_layers = n_layers