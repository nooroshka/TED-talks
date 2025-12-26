import os
import re
import requests
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

LLMOD_API_KEY = os.getenv("LLMOD_API_KEY")
LLMOD_BASE_URL = os.getenv("LLMOD_BASE_URL")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

TOP_K = int(os.getenv("TOP_K", "5"))

# LLMod embedding model id
EMBED_MODEL = "RPRTHPB-text-embedding-3-small"

# LLMod chat model id
CHAT_MODEL = "RPRTHPB-gpt-5-mini"

IDK = "I don't know based on the provided TED data."

if not all([LLMOD_API_KEY, LLMOD_BASE_URL, PINECONE_API_KEY, PINECONE_INDEX]):
    raise RuntimeError("Missing env vars. Check .env for LLMOD and PINECONE configs.")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)


def embed_text(text: str) -> list[float]:
    url = f"{LLMOD_BASE_URL}/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {LLMOD_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": EMBED_MODEL, "input": [text]}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def retrieve_context(question: str) -> list[dict]:
    q_emb = embed_text(question)
    res = index.query(vector=q_emb, top_k=TOP_K, include_metadata=True)

    hits = []
    for m in res.get("matches", []):
        meta = m.get("metadata", {}) or {}
        hits.append(
            {
                "score": float(m.get("score", 0.0)),
                "id": m.get("id"),
                "talk_id": meta.get("talk_id"),
                "title": meta.get("title"),
                "speaker": meta.get("speaker"),
                "chunk_index": meta.get("chunk_index"),
                "text": meta.get("text"),
            }
        )
    return hits


def build_augmented_prompt(question: str, hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(
            f"[{i}] talk_id={h.get('talk_id')} title={h.get('title')} speaker={h.get('speaker')} "
            f"chunk={h.get('chunk_index')} score={h.get('score')}\n"
            f"{h.get('text', '')}"
        )
    context = "\n\n".join(parts)

    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"TED CONTEXT (top {len(hits)} chunks):\n{context}\n\n"
        f"Answer the question using ONLY the TED CONTEXT.\n"
        f"Format: start with 'Title: ...' and 'Speaker: ...' when the question asks for title and speaker."
    )
    return user_prompt


def call_gpt(system_prompt: str, user_prompt: str) -> str:
    url = f"{LLMOD_BASE_URL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLMOD_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    r = requests.post(url, headers=headers, json=payload, timeout=90)
    if not r.ok:
        print("LLMod status:", r.status_code)
        print("LLMod response:", r.text)
        r.raise_for_status()

    return r.json()["choices"][0]["message"]["content"]


def answer_question(question: str, system_prompt: str) -> dict:
    hits = retrieve_context(question)
    augmented_user_prompt = build_augmented_prompt(question, hits)
    answer = call_gpt(system_prompt, augmented_user_prompt)


    if "I don't know based on the provided TED data." in answer:
        answer = IDK
    if "I don" in answer and "know based on the provided TED data" in answer:
        answer = IDK

    context = []
    for h in hits:
        context.append(
            {
                "talk_id": h.get("talk_id"),
                "title": h.get("title"),
                "chunk": h.get("text"),
                "score": float(h.get("score", 0.0)),
            }
        )

    return {
        "response": answer,
        "context": context,
        "Augmented_prompt": {
            "System": system_prompt,
            "User": augmented_user_prompt,
        },
    }
