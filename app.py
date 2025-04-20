"""
Gradio UI for Validated Patterns RAG LLM API
-------------------------------------------

Requirements::
    pip install gradio httpx python-dotenv

Environment variable::
    RAG_API_URL   # URL of vp-rag-llm-api backend (default: http://localhost:8000)

Run::
    export RAG_API_URL=http://your-backend:8000
    python rag_ui.py
"""

import os
import json
from typing import List, Dict, Generator, Optional

import httpx
import gradio as gr

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
API_URL = os.getenv("RAG_API_URL", "http://localhost:8080")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def get_models() -> tuple[list[str], Optional[str], dict]:
    """Return (choices, default_choice, grouped_models)."""
    try:
        resp = httpx.get(f"{API_URL}/models", timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print("⚠️  Could not fetch /models:", exc)
        return [], None, {}

    default_model = data.get("default_model")
    grouped: dict[str, list[str]] = {}
    for m in data.get("models", []):
        grouped.setdefault(m["model_type"], []).append(m["name"])

    choices = [f"{mtype}:{name}" for mtype, names in grouped.items() for name in names]
    default_choice = next(
        (c for c in choices if c.split(":", 1)[1] == default_model), None
    )
    return choices, default_choice, grouped


CHOICES, DEFAULT_CHOICE, _MODEL_GROUPS = get_models()


def _label_to_model(choice: Optional[str]) -> Optional[str]:
    """Strip the ``model_type:`` prefix back to the raw model name."""
    if not choice:
        return None
    return choice.split(":", 1)[1] if ":" in choice else choice


# -----------------------------------------------------------------------------
# Streaming client
# -----------------------------------------------------------------------------


def stream_chat(
    question: str, chat_history: list[Dict[str, str]], model_choice: Optional[str]
) -> Generator[str, None, None]:
    """Yield assistant tokens coming from the backend.

    ``chat_history`` follows the gradio **messages** schema: each item is
    ``{"role": "user"|"assistant", "content": "..."}``.
    The last two entries are the current user turn and an empty assistant
    placeholder; earlier entries are complete turns that will be sent as
    history.
    """

    # earlier fully‑formed messages (exclude current user + placeholder assistant)
    prev_messages = chat_history[:-2] if len(chat_history) > 1 else []
    history_flat = [m["content"] for m in prev_messages if m.get("content")]

    payload = {
        "question": question,
        "history": history_flat,
    }

    model = _label_to_model(model_choice)
    if model:
        payload["model"] = model

    with httpx.stream(
        "POST", f"{API_URL}/rag-query/stream", json=payload, timeout=None
    ) as r:
        for raw in r.iter_lines():  # iter_lines yields *str* by default (decoded)
            if not raw:
                continue
            if raw.startswith("data: "):
                data = raw[len("data: ") :]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue  # skip malformed lines
                token = chunk.get("content", "")
                yield token


# -----------------------------------------------------------------------------
# Gradio event handler
# -----------------------------------------------------------------------------


def respond(message: str, chat_history: list[Dict[str, str]], model_choice: str):
    """Streams assistant reply while updating the message list in‑place."""

    # Append current user message + empty assistant placeholder
    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]

    yield chat_history  # show user turn immediately

    assistant_buf = ""
    for token in stream_chat(message, chat_history, model_choice):
        assistant_buf += token
        chat_history[-1]["content"] = assistant_buf
        yield chat_history


# -----------------------------------------------------------------------------
# UI Layout
# -----------------------------------------------------------------------------
with gr.Blocks(title="Validated Patterns RAG Chat") as demo:
    gr.Markdown(
        """
        # Validated Patterns RAG Chat
        Enter a question about **Validated Patterns** or anything your RAG backend knows.
        _Example_: **How are secrets managed in Validated Patterns?**
        """
    )

    with gr.Row():
        model_sel = gr.Dropdown(
            choices=CHOICES,
            value=DEFAULT_CHOICE,
            label="Model (source:name)",
            interactive=True,
        )

    chatbot = gr.Chatbot(
        label="Chat", height=500, show_copy_button=True, type="messages"
    )
    msg = gr.Textbox(
        placeholder="How are secrets managed in Validated Patterns?",
        label="Your question",
        lines=1,
        autofocus=True,
    )

    clear = gr.Button("Clear")

    # Wire events
    msg.submit(respond, inputs=[msg, chatbot, model_sel], outputs=chatbot)
    clear.click(lambda: [], None, chatbot, queue=False)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch()
