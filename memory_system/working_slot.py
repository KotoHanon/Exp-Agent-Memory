from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union, Protocol

from .models import WorkingSnapshot
from .utils import new_id, dump_slot_json

from pydantic import BaseModel, Field, field_validator, validate_call

'''Dummy LLM for debugging'''
class LLMClient(Protocol):
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
    ...

class DummyLLM:
    def complete(self,
                 system_prompt: str, 
                 user_prompt: str, 
                 max_tokens: int = 512, 
                 temperature: float = 0.0
                 ) -> str:
        m = re.search(r"<slot-dump>\s*(\{.*\})\s*</slot-dump>", user_prompt, flags=re.S)
        slot = {}
        if m:
            try:
                slot = json.loads(m.group(1))
            except Exception:
                slot = {}
        summary = slot.get("summary", "") or " ".join([n.get("summary", "") for n in slot.get("attachments", {}).get("notes", [])])
        topic = slot.get("topic") or "task"
        has_metric = bool(slot.get("attachments", {}).get("metrics"))
        has_code = bool(slot.get("attachments", {}).get("code"))
        # Decide which tool prompt was asked by sniffing keywords
        if '"brief"' in user_prompt and '"keywords"' in user_prompt:
            brief = summary[:200] or f"{topic} update."
            out = {"brief": brief, "topic": topic, "keywords": list({w for w in re.findall(r"[A-Za-z0-9_]+", brief) if len(w) > 2})[:6]}
            return json.dumps(out, ensure_ascii=False)

        if '"type"' in user_prompt:
            t = "episodic" if has_metric else ("procedural" if has_code else "semantic")
            return t

        if '"cards"' in user_prompt:
            cards = []
            if has_metric:
                m0 = slot["attachments"]["metrics"][0]
                cards.append({"kind": "episode", "sar": {"S": topic, "A": "run", "R": f"{m0.get('name','metric')}={m0.get('value')}"}})
            if has_code:
                c0 = slot["attachments"]["code"][0]
                cards.append({"kind": "skill", "signature": "run(cmd)->result", "example": c0.get("snippet", "")[:120]})
            if summary:
                cards.append({"kind": "fact", "stmt": summary[:160]})
            if not cards:
                cards = [{"kind": "fact", "stmt": f"{topic} update."}]
            return json.dumps({"cards": cards[:3]}, ensure_ascii=False)

        if '"keywords"' in user_prompt and "retrieve" in user_prompt:
            words = list({w.lower() for w in re.findall(r"[A-Za-z0-9_]+", (summary or topic)) if len(w) > 2})
            return json.dumps({"keywords": words[:6] or [topic]}, ensure_ascii=False)

        return "{}"

class SlotPayload(BaseModel):
    stage: str = Field("", description="Stage of the working.")
    topic: str = Field("", description="Topic of the working slot.")
    summary: str = Field("", description="Summary of the working slot.")
    attachments: Dict[str, Dict] = Field(
        default_factory=list,
        description="List of attachment identifiers associated with the slot.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="List of tags associated with the slot.",
    )

class WorkingSlot(SlotPayload):
    """Transient context tied to the currently executing experiment."""

    def __init__(self, **kwargs) -> None:
        cfg = SlotPayload(**kwargs)

        self.id = new_id("work")
        self.stage = cfg.stage
        self.topic = cfg.topic
        self.summary = cfg.summary
        self.attachments = cfg.attachments
        self.tags = cfg.tags
    
    def slot_filter(self, llm: LLMClient) -> bool:
        system_prompt = "You are a memory access reviewer. Only output 'yes' or 'no'."
        user_prompt = f"""Determine whether this slot should be converted to long-term memory (LTM).

Evaluation dimensions: novelty (new information), utility (reusable value), stability (whether it is not easily outdated).

<slot-dump>
{dump_slot_json(self)}
</slot-dump>
"""
        out = llm.complete(system_prompt, user_prompt)
        return True if out.strip().lower() == "yes" else False
    
    def slot_router(self, llm: LLMClient) -> Literal["semantic", "procedural", "episodic"]:
        system_prompt = "You are a memory type classifier. Only output legal string."
        user_prompt = f"""Classify this slot into one of the following categories:

-semantic: General conclusions/rules that can be reused across tasks

-episodic: A certain process (S→A→R), including indicators/results

-procedural: Practices/steps/commands/function calls that can be reused as skills

Only output a string, either "semantic", "procedural", or "episodic".

<slot-dump>
{dump_slot_json(self)}
</slot-dump>
"""
        out = llm.complete(system_prompt, user_prompt)
        return out
