from GPT2.GPT2 import GPT2
from GPT2.conf.GPT2Config import GPT2Config

from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput


class GPT2Causal(PreTrainedModel, GenerationMixin):
    config_class = GPT2Config
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.gpt2 = GPT2(
            vocab_size=config.vocab_size,
            n_embed=config.n_embed,
            block_size=config.block_size,
            num_heads=config.num_heads,
            n_layers=config.n_layers,
        )

    def forward(self, input_ids, labels=None, **kwargs):
        logits, loss = self.gpt2(input_ids, targets=labels, flatten=False)
        return CausalLMOutput(logits=logits, loss=loss)
    
    def get_input_embeddings(self):
        return self.gpt2.embeddings

    def set_input_embeddings(self, value):
        self.gpt2.embeddings = value

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def _set_gradient_checkpointing(self, enable=True, gradient_checkpointing_func=None):
        # Propagate flag to all submodules that may use it
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = enable