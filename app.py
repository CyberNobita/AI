import gradio as gr
import os
from model_loader import MODELS, load_models
from chat_utils import chat, download_chat

# Load models at startup
loaded_models = load_models(MODELS)

# Get the port from the environment variable (Render-specific)
PORT = int(os.getenv("PORT", "8080"))

with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("<h1 style='text-align:center;'>💻 Lightweight Coding Chatbot</h1>")

    with gr.Row():
        model_selector = gr.Dropdown(choices=list(MODELS.keys()), value=list(MODELS.keys())[0], label="Choose Model")
        clear_button = gr.Button("🧹 Clear Chat")
        download_btn = gr.Button("⬇️ Download Chat")

    chatbot = gr.Chatbot(label="Conversation", show_copy_button=True)
    prompt_box = gr.Textbox(placeholder="Ask a coding question...", label="Your Prompt")
    submit_btn = gr.Button("🚀 Send")

    chat_state = gr.State([])

    submit_btn.click(
        lambda prompt, history, model: chat(prompt, history, model, loaded_models),
        inputs=[prompt_box, chat_state, model_selector],
        outputs=[chatbot, chat_state]
    )
    prompt_box.submit(
        lambda prompt, history, model: chat(prompt, history, model, loaded_models),
        inputs=[prompt_box, chat_state, model_selector],
        outputs=[chatbot, chat_state]
    )
    clear_button.click(lambda: ([], []), None, [chatbot, chat_state])
    download_btn.click(download_chat, [chat_state], None)

# Use the dynamic port for Render
demo.launch(server_name="0.0.0.0", server_port=PORT)
