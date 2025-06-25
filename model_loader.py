from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODELS = {
    "Flan-T5-Small": "google/flan-t5-small",
}

def load_models(model_map):
    print("Loading models...")
    all_models = {}
    for name, hf_id in model_map.items():
        tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            hf_id,
            torch_dtype=torch.float32
        )
        all_models[name] = {"tokenizer": tokenizer, "model": model}
    print("All models loaded.")
    return all_models
