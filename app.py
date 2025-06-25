import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODELS = {
    "Qwen3-0.6B": "rohitnagareddy/Qwen3-0.6B-Coding-Finetuned-v1",
    # Add more model names and IDs here
}

loaded_models = {}
for name, hf_id in MODELS.items():
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    loaded_models[name] = {"tokenizer": tokenizer, "model": model}

def chat(prompt, history, model_name):
    tokenizer = loaded_models[model_name]["tokenizer"]
    model = loaded_models[model_name]["model"]

    history.append(("🧑 " + prompt, ""))
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=256)
    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    history[-1] = (history[-1][0], "🤖 " + reply.strip())
    return history, history

def download_chat(chat_history):
    return "\n\n".join([f"{u}\n{b}" for u, b in chat_history])

with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("<h1 style='text-align:center;'>💻 Multi-Model Coding Assistant</h1>")

    with gr.Row():
        model_selector = gr.Dropdown(choices=list(MODELS.keys()), value=list(MODELS.keys())[0], label="Choose Model")
        clear_button = gr.Button("🧹 Clear Chat")
        download_btn = gr.Button("⬇️ Download Chat")

    chatbot = gr.Chatbot(label="Conversation", show_copy_button=True)
    prompt_box = gr.Textbox(placeholder="Ask to generate or explain code...", label="Your Prompt")
    submit_btn = gr.Button("🚀 Send")

    chat_state = gr.State([])

    submit_btn.click(chat, [prompt_box, chat_state, model_selector], [chatbot, chat_state])
    prompt_box.submit(chat, [prompt_box, chat_state, model_selector], [chatbot, chat_state])
    clear_button.click(lambda: ([], []), None, [chatbot, chat_state])
    download_btn.click(download_chat, [chat_state], file_name="chat.txt")

demo.launch(server_name="0.0.0.0", server_port=8080)
