import json
import os

import gradio as gr
import httpx

API_URL = os.getenv("RAG_API_URL", "http://localhost:8080")


async def fetch_models():
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{API_URL}/models")
        resp.raise_for_status()
        return await resp.json()


async def stream_rag(message, history, model):
    full_response = ""
    past_user_inputs = [turn[0] for turn in history if turn[0]]
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
                        delta = chunk.get("content", "")
                        if delta:
                            full_response += delta
                            yield full_response
                    except Exception as e:
                        print(f"⚠️ Failed to parse chunk: {e}\n{data}")

    if not full_response:
        yield "⚠️ No response generated. Please try rephrasing your question."


with gr.Blocks() as demo:
    gr.Markdown("## 🤖 OpenShift Pattern Assistant")
    gr.Markdown(
        "Ask questions about OpenShift Validated Patterns — usage, customization, testing, and more."
    )

    model_dropdown = gr.Dropdown(label="Select Model", interactive=True)

    chatbot = gr.Chatbot(render_markdown=True)
    msg = gr.Textbox(
        label="Ask a question...",
        placeholder="e.g. How are secrets managed in patterns?",
    )
    clear = gr.Button("Clear Chat")

    state = gr.State([])

    async def handle_submit(message, history, model_choice):
        async for partial in stream_rag(message, history, model_choice):
            yield partial

    msg.submit(fn=handle_submit, inputs=[msg, chatbot, model_dropdown], outputs=chatbot)
    clear.click(lambda: ([], []), outputs=[chatbot, state])

    # ⬇️ Populate models when UI loads
    async def populate_models():
        models = await fetch_models()
        return gr.update(choices=models["models"], value=models["default_model"])

    demo.load(fn=populate_models, outputs=model_dropdown)

demo.launch(server_name="0.0.0.0", server_port=7860)
