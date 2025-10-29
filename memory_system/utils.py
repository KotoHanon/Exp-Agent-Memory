from datetime import datetime, timezone
from uuid import uuid4
from typing import Iterable, Optional, Tuple

import json
import numpy as np


def now_iso() -> str:
    return datetime.now().isoformat()


def new_id(prefix: str) -> str:
    uuid_hex = uuid4().hex[:8]
    return f"{prefix}_{uuid_hex}"


def compute_overlap_score(text: str, query: str, keywords: Optional[Iterable[str]] = None) -> float:
    """Cheap lexical relevance score in [0, 1]."""
    if not text or not query:
        return 0.0
    text_lower = text.lower()
    query_lower = query.lower()
    overlap = sum(1 for word in query_lower.split() if word in text_lower)
    base_score = overlap / max(len(query_lower.split()), 1)
    if keywords:
        hit_bonus = sum(0.1 for keyword in keywords if keyword.lower() in text_lower)
    else:
        hit_bonus = 0.0
    return min(1.0, base_score + hit_bonus)


def ensure_tuple(value: Optional[Iterable]) -> Tuple:
    if value is None:
        return tuple()
    if isinstance(value, tuple):
        return value
    return tuple(value)


def _nomralize_embedding(emb: np.float32) -> np.float32:
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm 


def _jsonable_meta(meta: dict) -> dict:
    output = {}
    for k, v in meta.items():
        output[k] = v.to_dict()
    return output

def dump_slot_json(slot) -> str:
    payload = {
        "id": slot.id,
        "stage": slot.stage
        "topic": slot.topic,
        "summary": slot.summary,
        "attachments": slot.attachments,
        "tags": slot.tags
    }
    return json.dumps(payload, ensure_ascii=False)