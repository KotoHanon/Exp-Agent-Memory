from typing import Any, Dict, List, Optional


class EpisodicRecord(object):
    def __init__(
        self,
        id,  # type: str
        idea_id,  # type: str
        stage,  # type: str
        summary,  # type: str
        detail,  # type: Dict[str, Any]
        metrics=None,  # type: Optional[Dict[str, Any]]
        tags=None,  # type: Optional[List[str]]
        created_at="",  # type: str
    ):
        self.id = id
        self.idea_id = idea_id
        self.stage = stage
        self.summary = summary
        self.detail = detail or {}
        self.metrics = metrics or {}
        self.tags = tags or []
        self.created_at = created_at

    def to_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "id": self.id,
            "idea_id": self.idea_id,
            "stage": self.stage,
            "summary": self.summary,
            "detail": self.detail,
            "metrics": self.metrics,
            "tags": list(self.tags),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload):
        # type: (Dict[str, Any]) -> "EpisodicRecord"
        return cls(
            id=payload.get("id", ""),
            idea_id=payload.get("idea_id", ""),
            stage=payload.get("stage", ""),
            summary=payload.get("summary", ""),
            detail=payload.get("detail", {}),
            metrics=payload.get("metrics"),
            tags=payload.get("tags"),
            created_at=payload.get("created_at", ""),
        )


class SemanticRecord(object):
    def __init__(
        self,
        id,  # type: str
        summary,  # type: str
        detail,  # type: str
        source_ids=None,  # type: Optional[List[str]]
        tags=None,  # type: Optional[List[str]]
        confidence=0.5,  # type: float
        created_at="",  # type: str
        updated_at="",  # type: str
    ):
        self.id = id
        self.summary = summary
        self.detail = detail
        self.source_ids = source_ids or []
        self.tags = tags or []
        self.confidence = confidence
        self.created_at = created_at
        self.updated_at = updated_at

    def to_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "id": self.id,
            "summary": self.summary,
            "detail": self.detail,
            "source_ids": list(self.source_ids),
            "tags": list(self.tags),
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload):
        # type: (Dict[str, Any]) -> "SemanticRecord"
        return cls(
            id=payload.get("id", ""),
            summary=payload.get("summary", ""),
            detail=payload.get("detail", ""),
            source_ids=payload.get("source_ids"),
            tags=payload.get("tags"),
            confidence=payload.get("confidence", 0.5),
            created_at=payload.get("created_at", ""),
            updated_at=payload.get("updated_at", ""),
        )


class WorkingSnapshot(object):
    def __init__(
        self,
        goal=None,  # type: Optional[str]
        hypotheses=None,  # type: Optional[List[str]]
        plan_steps=None,  # type: Optional[List[str]]
        active_tools=None,  # type: Optional[List[str]]
        scratchpad=None,  # type: Optional[Dict[str, Any]]
    ):
        self.goal = goal
        self.hypotheses = hypotheses or []
        self.plan_steps = plan_steps or []
        self.active_tools = active_tools or []
        self.scratchpad = scratchpad or {}

    def to_dict(self):
        # type: () -> Dict[str, Any]
        return {
            "goal": self.goal,
            "hypotheses": list(self.hypotheses),
            "plan_steps": list(self.plan_steps),
            "active_tools": list(self.active_tools),
            "scratchpad": dict(self.scratchpad),
        }
