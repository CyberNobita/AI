from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODELS = {
    "Tiny-GPT2": "sshleifer/tiny-gpt2"
}

def load_models(model_map):
    all_models = {}
    for name, hf_id in model_map.items():
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch.float32,
        )
        model.eval()
        all_models[name] = {"tokenizer": tokenizer, "model": model}
    return all_models
