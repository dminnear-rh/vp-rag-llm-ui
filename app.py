import asyncio
import json
import os

import gradio as gr
import httpx

API_URL = os.getenv("RAG_API_URL", "http://localhost:8080")


# === Fetch available models from the API ===
async def fetch_models():
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{API_URL}/models")
        resp.raise_for_status()
        data = resp.json()
        return data["models"], data["default_model"]


# === RAG Streaming Function ===
async def stream_rag(message, history, model_name):
    full_response = ""
    payload = {
        "question": message,
        "history": [turn["content"] for turn in history if turn["role"] == "user"],
        "model": model_name,
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST", f"{API_URL}/rag-query/stream", json=payload
        ) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    data = line.removeprefix("data: ").strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        content = chunk.get("content", "")
                        full_response += content
                        yield full_response
                    except Exception as e:
                        yield f"⚠️ Failed to parse chunk: {e}\n{data}"


# === Main App Launcher ===
def launch_app():
    models_data, default_model = asyncio.run(fetch_models())
    model_names = [m["name"] for m in models_data]

    with gr.Blocks() as demo:
        with gr.Row():
            model_dropdown = gr.Dropdown(
                label="Select Model",
                choices=model_names,
                value=default_model,
                interactive=True,
            )

        chatbot = gr.Chatbot(
            label="🤖 OpenShift Pattern Assistant",
            render_markdown=True,
            type="messages",
        )
        message_input = gr.Textbox(
            placeholder="Ask a question...", label="Your message"
        )

        async def handle_chat(message, history):
            model = model_dropdown.value
            return await stream_rag(message, history, model)

        gr.ChatInterface(
            fn=handle_chat,
            chatbot=chatbot,
            textbox=message_input,
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    launch_app()
