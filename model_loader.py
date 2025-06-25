from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Add more Hugging Face model IDs here
MODELS = {
    "Qwen3-0.6B": "rohitnagareddy/Qwen3-0.6B-Coding-Finetuned-v1",
    # "GPT2": "gpt2",  # Optional extra
}

def load_models(model_map):
    print("Loading models...")
    all_models = {}
    for name, hf_id in model_map.items():
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        all_models[name] = {"tokenizer": tokenizer, "model": model}
    print("All models loaded.")
    return all_models
