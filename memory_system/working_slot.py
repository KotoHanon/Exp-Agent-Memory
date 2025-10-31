from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union, Protocol
from .models import WorkingSnapshot
from .utils import new_id, dump_slot_json
from pydantic import BaseModel, Field, field_validator, validate_call
from openai import OpenAI

class LLMClient(Protocol):
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        ) -> str:
        ...

class OpenAIClient:
    def __init__(self, model: str = "gpt-4.1-mini", client: Optional[OpenAI] = None) -> None:
        self._client = client or OpenAI()
        self._model = model

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        response = self._client.responses.create(
            model=self._model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        if getattr(response, "output_text", None):
            return response.output_text

        chunks: List[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "output_text":
                chunks.append(getattr(item, "text", ""))
            elif getattr(item, "content", None):
                # Fallback for older SDK payloads
                for part in item.content:
                    if part.get("type") == "output_text":
                        chunks.append(part.get("text", ""))
        return "".join(chunks)

class SlotPayload(BaseModel):
    id: str = Field(default_factory=lambda: new_id("work"))
    stage: str = Field("", description="Stage of the working.")
    topic: str = Field("", description="Topic of the working slot.")
    summary: str = Field("", description="Summary of the working slot.")
    attachments: Dict[str, Dict] = Field(
        default_factory=dict,
        description="List of attachment identifiers associated with the slot.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="List of tags associated with the slot.",
    )

class WorkingSlot(SlotPayload):
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "stage": self.stage,
            "topic": self.topic,
            "summary": self.summary,
            "attachments": self.attachments,
            "tags": self.tags,
        }
    
    def slot_filter(self, llm: LLMClient) -> bool:
        system_prompt = "You are a memory access reviewer. Only output 'yes' or 'no'."
        user_prompt = f"""Determine whether this slot should be converted to long-term memory (LTM).

Evaluation dimensions: novelty (new information), utility (reusable value), stability (whether it is not easily outdated).

<slot-dump>
{dump_slot_json(self)}
</slot-dump>
"""
        out = llm.complete(system_prompt, user_prompt)
        print(f"Slot filter output: {out}")
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
        if out.strip() not in ["semantic", "procedural", "episodic"]:
            raise ValueError(f"Invalid slot type: {out}")
        return out
