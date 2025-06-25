from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Lightweight model (~15MB) perfect for 512MB RAM
MODELS = {
    "Tiny-GPT2": "sshleifer/tiny-gpt2",
}

def load_models(model_map):
    print("Loading models...")
    all_models = {}
    for name, hf_id in model_map.items():
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch.float32,
            device_map=None
        )
        all_models[name] = {"tokenizer": tokenizer, "model": model}
    print("All models loaded.")
    return all_models
