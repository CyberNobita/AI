# app.py
import gradio as gr
from model_loader import MODELS, load_models
from chat_utils import chat, download_chat

loaded_models = load_models(MODELS)

with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("<h1 style='text-align:center;'>🧠 Lightweight Coding Assistant</h1>")

    with gr.Row():
        model_selector = gr.Dropdown(choices=list(MODELS.keys()), value=list(MODELS.keys())[0], label="Choose Model")
        clear_button = gr.Button("🧹 Clear Chat")
        download_btn = gr.Button("⬇️ Download Chat")

    chatbot = gr.Chatbot(label="Conversation", show_copy_button=True)
    prompt_box = gr.Textbox(placeholder="Ask a coding question...", label="Your Prompt")
    submit_btn = gr.Button("🚀 Send")

    chat_state = gr.State([])

    submit_btn.click(chat, [prompt_box, chat_state, model_selector, loaded_models], [chatbot, chat_state])
    prompt_box.submit(chat, [prompt_box, chat_state, model_selector, loaded_models], [chatbot, chat_state])
    clear_button.click(lambda: ([], []), None, [chatbot, chat_state])
    download_btn.click(download_chat, [chat_state], file_name="chat.txt")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
