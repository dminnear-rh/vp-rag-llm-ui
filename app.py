import os
import json
from typing import Dict, Generator, Optional

import httpx
import gradio as gr

API_URL = os.getenv("RAG_API_URL", "http://localhost:8080")

# -----------------------------------------------------------------------------
# Model helpers (lazy)
# -----------------------------------------------------------------------------


def fetch_models() -> tuple[list[str], Optional[str]]:
    """Hit /models and return (choices, default). Empty list on failure."""
    try:
        r = httpx.get(f"{API_URL}/models", timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        print("⚠️ Could not fetch /models:", exc)
        return [], None

    choices = [f"{m['model_type']}:{m['name']}" for m in data.get("models", [])]
    default = next(
        (c for c in choices if c.split(":", 1)[1] == data.get("default_model")),
        None,
    )
    return choices, default


def label_to_model(choice: str | None) -> str | None:
    return choice.split(":", 1)[1] if choice and ":" in choice else choice


# -----------------------------------------------------------------------------
# Streaming
# -----------------------------------------------------------------------------


def stream_chat(
    question: str, chat_history: list[Dict[str, str]], model_choice: str | None
) -> Generator[str, None, None]:
    prev = chat_history[:-2] if len(chat_history) >= 2 else []
    history_flat = [m["content"] for m in prev if m.get("content")]

    payload = {"question": question, "history": history_flat}
    model_name = label_to_model(model_choice)
    if model_name:
        payload["model"] = model_name

    with httpx.stream(
        "POST", f"{API_URL}/rag-query/stream", json=payload, timeout=None
    ) as r:
        for raw in r.iter_lines():
            if not raw or not raw.startswith("data: "):
                continue
            chunk = raw[6:]
            if chunk.strip() == "[DONE]":
                break
            try:
                token = json.loads(chunk)["content"]
            except (json.JSONDecodeError, KeyError):
                continue
            yield token


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------


def respond(message: str, chat_history: list[Dict[str, str]], model_choice: str):
    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    yield chat_history, ""  # clear textbox

    buf = ""
    for token in stream_chat(message, chat_history, model_choice):
        buf += token
        chat_history[-1]["content"] = buf
        yield chat_history, ""


def refresh_dropdown():
    """Return an update dict compatible with all Gradio versions."""
    choices, default = fetch_models()
    if not choices:
        return gr.update(
            choices=["⚠️ backend unreachable"], value=None, interactive=False
        )
    return gr.update(choices=choices, value=default, interactive=True)


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
with gr.Blocks(title="Validated Patterns RAG Chat", theme="soft") as demo:
    gr.Markdown(
        """
        # Validated Patterns RAG Chat
        _Ask about architecture, GitOps, Operators, pipelines, secret management…_
        """
    )

    with gr.Row():
        model_sel = gr.Dropdown(label="Model (source:name)", interactive=True, scale=5)
        refresh_btn = gr.Button("⟳", variant="secondary", scale=1)

    chatbot = gr.Chatbot(
        label="Conversation", height=500, show_copy_button=True, type="messages"
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="How are secrets managed in Validated Patterns?",
            lines=1,
            autofocus=True,
            show_label=False,
            container=False,
        )
        send_btn = gr.Button("Send", variant="primary")

    gr.Markdown("<small>Hit <kbd>Enter</kbd> or click <b>Send</b></small>")
    clear_btn = gr.Button("Clear chat")

    # Events
    demo.load(refresh_dropdown, None, model_sel)
    refresh_btn.click(refresh_dropdown, None, model_sel)

    for trg in (msg.submit, send_btn.click):
        trg(respond, inputs=[msg, chatbot, model_sel], outputs=[chatbot, msg])

    clear_btn.click(lambda: ([], ""), None, [chatbot, msg], queue=False)

if __name__ == "__main__":
    demo.launch()
