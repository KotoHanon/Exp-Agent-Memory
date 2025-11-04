import asyncio

from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union, Protocol
from .utils import new_id, dump_slot_json
from pydantic import BaseModel, Field, field_validator, validate_call
from openai import OpenAI
from textwrap import dedent
from .user_prompt import WORKING_SLOT_FILTER_USER_PROMPT, WORKING_SLOT_ROUTE_USER_PROMPT

class LLMClient(Protocol):
    async def complete(
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

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Optional[Exception] = None

        try:
            response = await asyncio.to_thread(
                self._client.responses.create,
                model=self._model,
                input=messages,
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            if hasattr(response, "output_text"):
                return response.output_text
        except (AttributeError, TypeError) as exc:
            last_error = exc

        try:
            response = await asyncio.to_thread(
                self._client.chat.completions.create,
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            message = response.choices[0].message
            return message["content"] if isinstance(message, dict) else message.content
        except AttributeError as exc:
            last_error = last_error or exc

        try:
            response = await asyncio.to_thread(
                self._client.ChatCompletion.create,
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message["content"]
        except Exception as exc:
            raise last_error or exc

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
    
    async def slot_filter(self, llm: LLMClient) -> bool:
        system_prompt = "You are a memory access reviewer. Only output 'yes' or 'no'."
        user_prompt = WORKING_SLOT_FILTER_USER_PROMPT.format(slot_dump=dump_slot_json(self))
        out = await llm.complete(system_prompt, user_prompt)

        if out.strip().lower() not in ["yes", "no"]:
            raise ValueError(f"Invalid slot filter output: {out}")

        return True if out.strip().lower() == "yes" else False
    
    async def slot_router(self, llm: LLMClient) -> Literal["semantic", "procedural", "episodic"]:
        system_prompt = "You are a memory type classifier. Only output legal string."
        user_prompt = WORKING_SLOT_ROUTE_USER_PROMPT.format(slot_dump=dump_slot_json(self))
        out = await llm.complete(system_prompt, user_prompt)
        if out.strip() not in ["semantic", "procedural", "episodic"]:
            raise ValueError(f"Invalid slot type: {out}")
        return out
