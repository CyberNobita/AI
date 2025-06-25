import tempfile

def chat(prompt, history, model_name, loaded_models):
    # Retrieve the tokenizer and model
    tokenizer = loaded_models[model_name]["tokenizer"]
    model = loaded_models[model_name]["model"]

    # Ensure the model is on the CPU
    model = model.to("cpu")

    # Append the user's prompt to the chat history
    history.append(("🧑 " + prompt, ""))

    # Tokenize the input and generate a response
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")  # Ensure inputs are on CPU
    output = model.generate(**inputs, max_new_tokens=128)
    reply = tokenizer.decode(output[0], skip_special_tokens=True)

    # Append the model's reply to the chat history
    history[-1] = (history[-1][0], "🤖 " + reply.strip())
    return history, history

def download_chat(chat_history):
    try:
        # Create a temporary file to store the chat history
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as f:
            for user, bot in chat_history:
                f.write(f"{user}\n{bot}\n\n")
            return f.name  # Return the file path for Gradio's File component
    except Exception as e:
        # Handle file creation errors
        return f"[Error] Failed to create chat history file: {e}"
