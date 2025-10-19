from typing import Dict, List, Optional

from .models import WorkingSnapshot
from .utils import now_iso


class WorkingMemory:
    """Transient context tied to the currently executing experiment."""

    def __init__(self) -> None:
        self._history: List[WorkingSnapshot] = []
        self._snapshot = WorkingSnapshot()
        self._last_updated = now_iso()

    def update_goal(self, goal: str) -> None:
        self._snapshot.goal = goal
        self._mark_updated()

    def add_hypothesis(self, hypothesis: str) -> None:
        self._snapshot.hypotheses.append(hypothesis)
        self._mark_updated()

    def set_plan(self, steps: List[str]) -> None:
        self._snapshot.plan_steps = list(steps)
        self._mark_updated()

    def add_active_tool(self, tool: str) -> None:
        if tool not in self._snapshot.active_tools:
            self._snapshot.active_tools.append(tool)
            self._mark_updated()

    def write_scratchpad(self, key: str, value: object) -> None:
        self._snapshot.scratchpad[key] = value
        self._mark_updated()

    def freeze(self) -> WorkingSnapshot:
        snapshot = WorkingSnapshot(
            goal=self._snapshot.goal,
            hypotheses=list(self._snapshot.hypotheses),
            plan_steps=list(self._snapshot.plan_steps),
            active_tools=list(self._snapshot.active_tools),
            scratchpad=dict(self._snapshot.scratchpad),
        )
        self._history.append(snapshot)
        self._mark_updated()
        return snapshot

    def latest(self) -> WorkingSnapshot:
        return self._snapshot

    def history(self) -> List[WorkingSnapshot]:
        return list(self._history)

    def summary(self) -> Dict[str, Optional[str]]:
        latest_plan = self._snapshot.plan_steps[:2] if self._snapshot.plan_steps else []
        return {
            "goal": self._snapshot.goal,
            "plan_preview": " | ".join(latest_plan),
            "updated_at": self._last_updated,
        }

    def _mark_updated(self) -> None:
        self._last_updated = now_iso()
