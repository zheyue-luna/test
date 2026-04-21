import os
import random
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock

import genanki
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field


class Flashcard(BaseModel):
    front: str = Field(description="Question, term, or prompt shown on the front of the card.")
    back: str = Field(description="Concise, complete answer or explanation shown on the back.")
    tags: list[str] = Field(
        default_factory=list,
        description="1-3 short tags for categorization. No spaces.",
    )


class FlashcardDeck(BaseModel):
    deck_name: str = Field(description="A concise, descriptive deck title (max 8 words).")
    cards: list[Flashcard] = Field(description="Extracted flashcards.")


class GenerateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200_000)
    deck_name: str | None = None
    language: str | None = None
    max_cards: int | None = Field(default=None, ge=1, le=200)


class GenerateResponse(BaseModel):
    download_url: str
    deck_name: str
    card_count: int
    expires_at: str


APKG_DIR = Path(tempfile.gettempdir()) / "anki_apkg"
APKG_DIR.mkdir(exist_ok=True)
FILE_TTL_SECONDS = 3600

_registry: dict[str, dict] = {}
_registry_lock = Lock()

OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
app = FastAPI(title="Anki Flashcard Generator")

SYSTEM_PROMPT = """You extract high-quality Anki flashcards from arbitrary text.

Principles:
- One atomic fact per card (minimum information principle).
- Front: a specific question, term, or cloze prompt. Back: a concise, complete answer.
- Cover definitions, key facts, relationships, formulas, causes/effects — skip trivia and filler.
- Preserve the source language (if input is Chinese, output cards in Chinese).
- No duplicate or near-duplicate cards.
- Provide a short descriptive deck_name and 1-3 tags per card (lowercase, no spaces, use underscores)."""


ANKI_MODEL = genanki.Model(
    1607392319,
    "AI Basic",
    fields=[{"name": "Front"}, {"name": "Back"}],
    templates=[
        {
            "name": "Card 1",
            "qfmt": "{{Front}}",
            "afmt": '{{FrontSide}}<hr id="answer">{{Back}}',
        }
    ],
    css=(
        ".card{font-family:-apple-system,BlinkMacSystemFont,'PingFang SC',"
        "'Microsoft YaHei',sans-serif;font-size:20px;line-height:1.5;"
        "text-align:center;color:#222;background:#fff;padding:20px;}"
    ),
)


def extract_flashcards(req: GenerateRequest) -> FlashcardDeck:
    hints = []
    if req.deck_name:
        hints.append(f"Preferred deck name: {req.deck_name}")
    if req.language:
        hints.append(f"Output language: {req.language}")
    if req.max_cards:
        hints.append(f"Produce at most {req.max_cards} cards.")
    hint_block = ("\n" + "\n".join(hints)) if hints else ""

    user_prompt = (
        f"Extract Anki flashcards from the following source text.{hint_block}\n\n"
        f"<source>\n{req.text}\n</source>"
    )

    response = client.beta.chat.completions.parse(
        model=OPENROUTER_MODEL,
        max_tokens=16000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=FlashcardDeck,
    )
    deck = response.choices[0].message.parsed
    if deck is None:
        raise HTTPException(status_code=502, detail="Model did not return a parseable flashcard set.")
    return deck


def build_apkg(deck_data: FlashcardDeck, output_path: Path) -> int:
    deck = genanki.Deck(random.randrange(1 << 30, 1 << 31), deck_data.deck_name)
    for card in deck_data.cards:
        deck.add_note(
            genanki.Note(
                model=ANKI_MODEL,
                fields=[card.front, card.back],
                tags=[t.replace(" ", "_") for t in card.tags],
            )
        )
    genanki.Package(deck).write_to_file(str(output_path))
    return len(deck_data.cards)


def cleanup_expired() -> None:
    now = time.time()
    with _registry_lock:
        expired = [t for t, m in _registry.items() if m["expires_at"] < now]
        for token in expired:
            meta = _registry.pop(token)
            Path(meta["path"]).unlink(missing_ok=True)


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, bg: BackgroundTasks) -> GenerateResponse:
    bg.add_task(cleanup_expired)

    try:
        deck_data = extract_flashcards(req)
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"Model error: {e}") from e

    if not deck_data.cards:
        raise HTTPException(status_code=422, detail="No flashcards could be extracted from the text.")

    token = uuid.uuid4().hex
    out_path = APKG_DIR / f"{token}.apkg"
    count = build_apkg(deck_data, out_path)
    expires_at = time.time() + FILE_TTL_SECONDS

    with _registry_lock:
        _registry[token] = {
            "path": str(out_path),
            "deck_name": deck_data.deck_name,
            "expires_at": expires_at,
        }

    return GenerateResponse(
        download_url=f"/api/download/{token}",
        deck_name=deck_data.deck_name,
        card_count=count,
        expires_at=datetime.fromtimestamp(expires_at).isoformat(),
    )


@app.get("/api/download/{token}")
def download(token: str) -> FileResponse:
    with _registry_lock:
        meta = _registry.get(token)
        if not meta:
            raise HTTPException(status_code=404, detail="Not found or expired.")
        if meta["expires_at"] < time.time():
            _registry.pop(token, None)
            Path(meta["path"]).unlink(missing_ok=True)
            raise HTTPException(status_code=410, detail="File expired.")
        path = meta["path"]
        deck_name = meta["deck_name"]

    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in deck_name).strip() or "deck"
    return FileResponse(
        path,
        media_type="application/apkg",
        filename=f"{safe[:60]}.apkg",
    )


@app.get("/")
def root() -> dict:
    return {
        "service": "Anki Flashcard Generator",
        "endpoints": {
            "generate": "POST /api/generate",
            "download": "GET /api/download/{token}",
        },
        "file_ttl_seconds": FILE_TTL_SECONDS,
    }
