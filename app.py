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
from typing import List, Tuple, Generator, Optional

import httpx
import gradio as gr

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
API_URL = os.getenv("RAG_API_URL", "http://localhost:8080")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def get_models() -> Tuple[List[str], Optional[str], dict]:
    """Return (choices, default_choice, grouped_models).

    * ``choices`` is a flat list suitable for gr.Dropdown (e.g. ["openai:gpt-4o", "vllm:mistral-7b"]).
    * Each label is prefixed with the model_type so users know where it lives.
    * ``default_choice`` is the choice label that matches the API's default model.
    * ``grouped_models`` preserves the grouping in case you want to build a custom
      component later.
    """
    try:
        resp = httpx.get(f"{API_URL}/models", timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print("⚠️  Could not fetch /models:", exc)
        return [], None, {}

    default_model = data.get("default_model")
    grouped = {}
    for m in data.get("models", []):
        grouped.setdefault(m["model_type"], []).append(m["name"])

    choices = [f"{mtype}:{name}" for mtype, names in grouped.items() for name in names]
    default_choice = next(
        (c for c in choices if c.split(":", 1)[1] == default_model), None
    )
    return choices, default_choice, grouped


CHOICES, DEFAULT_CHOICE, MODEL_GROUPS = get_models()


def _label_to_model(choice: Optional[str]) -> Optional[str]:
    """Strip the ``model_type:`` prefix back to the raw model name."""
    if not choice:
        return None
    return choice.split(":", 1)[1] if ":" in choice else choice


# -----------------------------------------------------------------------------
# Streaming client
# -----------------------------------------------------------------------------


def stream_chat(
    question: str, chat_history: List[Tuple[str, str]], model_choice: Optional[str]
) -> Generator[str, None, None]:
    """Generator that yields incremental assistant tokens coming from the backend.

    ``chat_history`` is a list of (user, assistant) tuples *including* the current
    turn whose assistant part is still empty. We send **all fully‑formed previous
    messages** (both user and assistant) back to the backend so it has full
    conversational context.
    """

    # build flattened history excluding the unfinished last turn
    prev_pairs = chat_history[:-1]
    history_flat: List[str] = []
    for user_msg, assistant_msg in prev_pairs:
        if user_msg:
            history_flat.append(user_msg)
        if assistant_msg:
            history_flat.append(assistant_msg)

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
        for raw in r.iter_lines():
            if not raw:
                continue
            if raw.startswith(b"data: "):
                data = raw[len(b"data: ") :].decode()
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    # Skip malformed lines just in case
                    continue
                token = chunk.get("content", "")
                yield token


# -----------------------------------------------------------------------------
# Gradio event handler
# -----------------------------------------------------------------------------


def respond(message: str, chat_history: List[Tuple[str, str]], model_choice: str):
    """Adds the new user turn, streams assistant tokens, and updates UI."""
    chat_history = chat_history + [(message, "")]  # append placeholder for assistant
    yield chat_history  # display user message immediately

    assistant_buffer = ""
    for token in stream_chat(message, chat_history, model_choice):
        assistant_buffer += token
        chat_history[-1] = (message, assistant_buffer)
        # continually update the UI so users see tokens in real time
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

    chatbot = gr.Chatbot(label="Chat", height=500, show_copy_button=True)
    msg = gr.Textbox(
        placeholder="How are secrets managed in Validated Patterns?",
        label="Your question",
        lines=1,
        autofocus=True,
    )

    clear = gr.Button("Clear")

    # Wire events
    msg.submit(respond, inputs=[msg, chatbot, model_sel], outputs=chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch()
