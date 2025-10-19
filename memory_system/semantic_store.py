from typing import Iterable, List, Optional
import faiss
import numpy as np

from .models import SemanticRecord
from .storage import JsonFileStore
from .utils import compute_overlap_score, ensure_tuple, new_id, now_iso


class SemanticMemory:
    """Distilled insights usable across experiments."""

    def __init__(self, path: str = "memory_data/semantic.json") -> None:
        self._store = JsonFileStore(path)

    def add(
        self,
        summary: str,
        detail: str,
        source_ids: Optional[Iterable[str]] = None,
        tags: Optional[Iterable[str]] = None,
        confidence: float = 0.6,
    ) -> SemanticRecord:
        record = SemanticRecord(
            id=new_id("sem"),
            summary=summary,
            detail=detail,
            source_ids=list(source_ids or []),
            tags=list(tags or []),
            confidence=confidence,
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        self._store.append(record.to_dict())
        return record

    def update(
        self,
        record_id: str,
        summary: Optional[str] = None,
        detail: Optional[str] = None,
        confidence: Optional[float] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> SemanticRecord:
        items = self._store.load_all()
        updated = None  # type: Optional[SemanticRecord]
        for raw in items:
            if raw["id"] == record_id:
                record = SemanticRecord.from_dict(raw)
                if summary is not None:
                    record.summary = summary
                if detail is not None:
                    record.detail = detail
                if confidence is not None:
                    record.confidence = confidence
                if tags is not None:
                    record.tags = list(tags)
                record.updated_at = now_iso()
                updated = record
                break
        if updated is None:
            raise ValueError(f"Insight {record_id} not found.")
        self._store.update(record_id, updated.to_dict())
        return updated

    def list_recent(self, limit: int = 5) -> List[SemanticRecord]:
        items = self._store.load_all()
        subset = items[-limit:]
        return [SemanticRecord.from_dict(item) for item in subset]

    def query(
        self,
        query: str = "",
        tags: Optional[Iterable[str]] = None,
        min_confidence: float = 0.0,
        limit: int = 5,
    ) -> List[SemanticRecord]:
        matches = []
        tags = ensure_tuple(tags)
        for raw in reversed(self._store.load_all()):
            if raw.get("confidence", 0.0) < min_confidence:
                continue
            if tags and not set(tag.lower() for tag in raw.get("tags", [])).issuperset(
                {tag.lower() for tag in tags}
            ):
                continue
            score = compute_overlap_score(raw.get("summary", ""), query, tags)
            bonus = raw.get("confidence", 0.0)
            matches.append((score + bonus * 0.1, raw))

        matches.sort(key=lambda item: item[0], reverse=True)
        return [SemanticRecord.from_dict(raw) for _, raw in matches[:limit]]
