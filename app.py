import os
import json
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

    # Format history into just prior user messages
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
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            full_response += delta
                            yield full_response
                    except Exception as e:
                        print(f"⚠️ Failed to parse chunk: {e}\n{data}")

    if not full_response:
        yield "⚠️ No response generated. Please try rephrasing your question."


# === Build UI dynamically after fetching models ===
def launch_app():
    import asyncio

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

        chatbot = gr.Chatbot(render_markdown=True)
        msg = gr.Textbox(
            label="Ask a question...",
            placeholder="e.g. How are secrets managed in patterns?",
        )
        send_btn = gr.Button("Send")

        async def handle_submit(message, history, model_choice):
            async for partial in stream_rag(message, history, model_choice):
                yield partial

        send_btn.click(
            fn=handle_submit,
            inputs=[msg, chatbot, model_dropdown],
            outputs=chatbot,
        )
        msg.submit(
            fn=handle_submit,
            inputs=[msg, chatbot, model_dropdown],
            outputs=chatbot,
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    launch_app()
