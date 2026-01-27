from transformers import AutoConfig, AutoModelForCausalLM


from GPT2.conf.GPT2Config import GPT2Config
from GPT2.GPT2_causal import GPT2Causal

AutoConfig.register("my_gpt", GPT2Config, exist_ok=True)
AutoModelForCausalLM.register(GPT2Config, GPT2Causal, exist_ok=True)