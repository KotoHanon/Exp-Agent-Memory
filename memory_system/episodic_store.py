from typing import Iterable, List, Optional

from .models import EpisodicRecord
from .storage import JsonFileStore
from .utils import compute_overlap_score, new_id, now_iso, ensure_tuple


class EpisodicMemory:
    """Append-only log of experiment execution traces."""

    def __init__(self, path: str = "memory_data/episodic.json") -> None:
        self._store = JsonFileStore(path)

    def add(
        self,
        idea_id: str,
        stage: str,
        summary: str,
        detail: dict,
        metrics: Optional[dict] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> EpisodicRecord:
        record = EpisodicRecord(
            id=new_id("epi"),
            idea_id=idea_id,
            stage=stage,
            summary=summary,
            detail=detail,
            metrics=metrics or {},
            tags=list(tags or []),
            created_at=now_iso(),
        )
        self._store.append(record.to_dict())
        return record

    def list_recent(self, limit: int = 10) -> List[EpisodicRecord]:
        all_items = self._store.load_all()
        subset = all_items[-limit:]
        return [EpisodicRecord.from_dict(item) for item in subset]

    def query(
        self,
        query: str = "",
        idea_id: Optional[str] = None,
        stages: Optional[Iterable[str]] = None,
        tags: Optional[Iterable[str]] = None,
        limit: int = 5,
    ) -> List[EpisodicRecord]:
        all_items = self._store.load_all()
        stages = ensure_tuple(stages)
        tags = ensure_tuple(tags)

        matches = []
        for raw in reversed(all_items):
            if idea_id and raw["idea_id"] != idea_id:
                continue
            if stages and raw["stage"] not in stages:
                continue
            if tags and not tags_intersect(raw.get("tags", []), tags):
                continue
            score = compute_overlap_score(raw.get("summary", ""), query, tags)
            if score == 0.0 and query:
                continue
            matches.append((score, raw))

        matches.sort(key=lambda item: item[0], reverse=True)
        return [EpisodicRecord.from_dict(item[1]) for item in matches[:limit]]


def tags_intersect(record_tags: Iterable[str], target_tags: Iterable[str]) -> bool:
    record_lower = {item.lower() for item in record_tags}
    target_lower = {item.lower() for item in target_tags}
    return not record_lower.isdisjoint(target_lower)
