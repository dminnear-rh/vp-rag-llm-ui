import asyncio
import json
import os

import gradio as gr
import httpx

API_URL = os.getenv("RAG_API_URL", "http://localhost:8080")


async def fetch_models():
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{API_URL}/models")
        resp.raise_for_status()
        data = resp.json()
        return data["models"], data["default_model"]


async def stream_rag(message, history, model):
    full_response = ""

    # Reformat history for backend: extract only user messages
    past_user_inputs = [msg["content"] for msg in history if msg["role"] == "user"]

    payload = {"question": message, "history": past_user_inputs, "model": model}

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
                        delta = chunk["content"]
                        if delta:
                            full_response += delta
                            yield history + [
                                {"role": "user", "content": message},
                                {"role": "assistant", "content": full_response},
                            ]
                    except Exception as e:
                        print(f"⚠️ Failed to parse chunk: {e}\n{data}")

    if not full_response:
        yield history + [
            {"role": "user", "content": message},
            {
                "role": "assistant",
                "content": "⚠️ No response generated. Please try rephrasing your question.",
            },
        ]


def launch_app():
    models_data, default_model = asyncio.run(fetch_models())
    model_names = [m["name"] for m in models_data]

    with gr.Blocks() as demo:
        gr.Markdown("## 🤖 OpenShift Pattern Assistant")
        gr.Markdown(
            "Ask questions about OpenShift Validated Patterns — usage, customization, testing, and more."
        )

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=model_names,
                value=default_model,
                label="Select Model",
                interactive=True,
            )

        chatbot = gr.Chatbot(label="Chat", render_markdown=True, type="messages")
        msg = gr.Textbox(
            label="Ask a question...",
            placeholder="e.g. How are secrets managed in patterns?",
        )
        send_btn = gr.Button("Send")

        def on_submit(message, history, model_choice):
            return stream_rag(message, history, model_choice)

        send_btn.click(
            fn=on_submit,
            inputs=[msg, chatbot, model_dropdown],
            outputs=chatbot,
        )
        msg.submit(
            fn=on_submit,
            inputs=[msg, chatbot, model_dropdown],
            outputs=chatbot,
        )

    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    launch_app()
