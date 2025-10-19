from typing import Dict, Iterable, Optional

from .episodic_store import EpisodicMemory
from .semantic_store import SemanticMemory
from .working_memory import WorkingMemory


class MemoryRetriever:
    def __init__(
        self,
        working: WorkingMemory,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
    ) -> None:
        self.working = working
        self.episodic = episodic
        self.semantic = semantic

    def build_context(
        self,
        query: str,
        idea_id: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        episodic_limit: int = 3,
        semantic_limit: int = 3,
    ) -> Dict[str, object]:
        episodic = self.episodic.query(
            query=query,
            idea_id=idea_id,
            tags=tags,
            limit=episodic_limit,
        )
        semantic = self.semantic.query(
            query=query,
            tags=tags,
            min_confidence=0.4,
            limit=semantic_limit,
        )
        return {
            "working_memory": self.working.latest().to_dict(),
            "episodic": [record.to_dict() for record in episodic],
            "semantic": [record.to_dict() for record in semantic],
        }

    def plan_inputs(self, idea_id: str, query: str) -> Dict[str, object]:
        bundle = self.build_context(
            query=query,
            idea_id=idea_id,
            tags=["planning"],
            episodic_limit=5,
            semantic_limit=5,
        )
        bundle["working_summary"] = self.working.summary()
        return bundle
