# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import json

from app.rag import answer_question, TOP_K

app = FastAPI()

SYSTEM_PROMPT = """You are a TED Talk assistant that answers questions strictly and
only based on the TED dataset context provided to you (metadata
and transcript passages). You must not use any external
knowledge, the open internet, or information that is not explicitly
contained in the retrieved context. If the answer cannot be
determined from the provided context, respond: "I don't know
based on the provided TED data." Always explain your answer
using the given context, quoting or paraphrasing the relevant
transcript or metadata when helpful."""



class PromptRequest(BaseModel):
    question: str


def load_config() -> dict:
    # project_root is the folder that contains config.json and the app folder
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"config.json not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/prompt")
def prompt(req: PromptRequest):
    # Must return: response, context array, Augmented_prompt object (exact keys)
    return answer_question(req.question, SYSTEM_PROMPT)


@app.get("/api/stats")
def stats():
    cfg = load_config()
    chunk_size = int(cfg.get("chunk_size", 0))
    overlap_ratio = float(cfg.get("overlap_ratio", 0.0))

    return {
        "chunk_size": chunk_size,
        "overlap_ratio": overlap_ratio,
        "top_k": int(TOP_K),
    }
