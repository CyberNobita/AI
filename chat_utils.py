def chat(prompt, history, model_name, loaded_models):
    tokenizer = loaded_models[model_name]["tokenizer"]
    model = loaded_models[model_name]["model"]

    history.append(("🧑 " + prompt, ""))
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=256)
    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    history[-1] = (history[-1][0], "🤖 " + reply.strip())
    return history, history

def download_chat(chat_history):
    return "\n\n".join([f"{user}\n{bot}" for user, bot in chat_history])
