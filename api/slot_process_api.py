from memory_system import WorkingSlot, OpenAIClient, LLMClient
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union
from collections import deque
from memory_system.utils import dump_slot_json, _extract_json_between, _hard_validate_slot_keys
from textwrap import dedent

class SlotProcess:
    def __init__(self):
        self.slot_container: Dict[str, WorkingSlot] = {}
        self.filtered_slot_container: List[WorkingSlot] = []
        self.routed_slot_container: List[Dict] = []
        self.llm_model = OpenAIClient()

    def add_slot(self, slot: WorkingSlot) -> None:
        self.slot_container[slot.to_dict().get('id')] = slot
    
    def clear_container(self) -> None:
        self.slot_container = {}

    def get_container_size(self) -> int:
        return len(self.slot_container)
    
    async def filter_and_route_slots(self) -> deque[Dict[str, WorkingSlot]]:
        for slot in self.slot_container.values():
            check_result = await slot.slot_filter(self.llm_model)
            print(check_result)
            if check_result == True:
                self.filtered_slot_container.append(slot)
        
        try:
            for filtered_slot in self.filtered_slot_container:
                route_result = await filtered_slot.slot_router(self.llm_model)
                pair = {
                    "memory_type": route_result,
                    "slot": filtered_slot
                }
                self.routed_slot_container.append(pair)
        except Exception as e:
            print(f"Routing error: {e}")
        
        return self.routed_slot_container
    
    async def compress_slots(self, sids: List[str] = None) -> WorkingSlot:
        slot_json_blobs = []
        if sids is None:
            for idx, slot in enumerate(self.slot_container.values()):
                slot_json_blobs.append(f"### Slot {idx}\n{dump_slot_json(slot)}")
        else:
            for idx, slot_id in enumerate(sids):
                slot_json_blobs.append(f"### Slot {idx}\n{dump_slot_json(self.slot_container[slot_id])}")
        slots_block = "\n\n".join(slot_json_blobs)

        system_prompt = (
            "You are an expert research assistant and memory compressor. "
            "Given multiple WorkingSlot JSON dumps, produce a single, compact summary "
            "that preserves non-redundant, reusable knowledge while discarding noise. "
            "Be precise, consistent, and avoid hallucinations. Output only the requested JSON inside the tags."
        )

        user_prompt = dedent(f"""
                        Read the following WorkingSlots and compress them into ONE consolidated WorkingSlot.

                        Your goals:
                        - Deduplicate overlapping content while keeping key facts.
                        - Prefer information that is novel, useful across tasks, and stable (unlikely to expire soon).
                        - Preserve crucial metrics, decisions, and actionable steps when available.
                        - Resolve contradictions conservatively; if uncertain, prefer safer, broadly valid statements.
                        - Keep the final summary ≤ 150 words, concise and specific.

                        Input WorkingSlots (JSON):

                        <slots>
                        {slots_block}
                        </slots>

                        Output format (STRICTLY JSON wrapped by tags; ONLY these keys: stage, topic, summary, attachments, tags):
                        <compressed-slot>
                        {{
                        "stage": "compressed",                      // str
                        "topic": "a short topic slug",              // str
                        "summary": "≤150-word compact synthesis",   // str
                        "attachments": {{                           // dict[str, dict]; aggregate from inputs (do not invent)
                            "notes": {{"items": ["bullet1", "bullet2"]}},
                            "metrics": {{"acc": 0.91}},               // include only if present in inputs; do NOT fabricate
                            "procedures": {{"steps": ["step1", "step2"]}},
                            "sources": {{"ids": ["src1","src2"]}}
                        }},
                        "tags": ["tag1","tag2","tag3"]              // List[str]
                        }}
                        </compressed-slot>

                        STRICT CONTRACT:
                        - Top-level keys MUST be exactly: stage, topic, summary, attachments, tags.
                        - attachments MUST be a JSON object (dictionary).
                        - Attachment KEYS are not fixed; include ONLY keys that appear in the inputs (the union of observed keys). Do NOT invent new keys.
                        - For every attachment entry, the VALUE MUST be a JSON object (dictionary). Never output scalars or arrays at this level.

                        ATTACHMENT NORMALIZATION RULES:
                        - If an input attachment value is an object → shallow-merge objects across slots (later evidence refines earlier).
                        - If an input attachment value is an array → output {{"items": [...]}} (deduplicated, concise).
                        - If an input attachment value is a scalar (string/number/bool) → output {{"value": <scalar>}}.
                        - If multiple scalar values conflict → prefer the most recent slot; if unclear, pick the most conservative or omit.
                        - Numeric metrics MUST come only from inputs (no fabrication). If present under different keys, keep each under its original key.

                        OTHER RULES:
                        - summary ≤ 150 words, factual, reusable, and stable (avoid ephemeral details).
                        - tags is a short, relevant LIST (≤ 10), deduplicated.
                        - Do NOT include any other top-level keys. Do NOT output nulls; omit missing fields inside attachments instead.
                        - Output STRICT JSON only, wrapped in the required tags.
                        """)

        response = await self.llm_model.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        payload = _extract_json_between(response, "compressed-slot", "compressed-slot")
        print(f"Compressed slot payload: {payload}")
        try:
            _hard_validate_slot_keys(payload, allowed_keys={"stage", "topic", "summary", "attachments", "tags"})
        except Exception as e:
            raise ValueError(f"Compressed slot validation error: {e}")
        
        stage = str(payload.get("stage", ""))
        topic = str(payload.get("topic", ""))
        summary = str(payload.get("summary", ""))
        attachments = payload.get("attachments", {})
        tags = payload.get("tags", [])

        compressed_slot = WorkingSlot(
            stage=stage,
            topic=topic,
            summary=summary,
            attachments=attachments,
            tags=tags
        )

        return compressed_slot
    
    async def transfer_slot_to_text(self, slot: WorkingSlot) -> str:
        system_prompt = (
            "You are an expert assistant that converts WorkingSlot JSON data into a clear, concise text summary. "
            "Focus on key insights, important metrics, and actionable items. Output only the requested text inside the tags."
        )

        user_prompt = dedent(f"""
                        Convert the following WorkingSlot JSON into a concise text summary.

                        Input WorkingSlot (JSON):

                        {dump_slot_json(slot)}

                        Output format: A plain text, WITHOUT ANY WRAPTAG.
                        [Your concise text summary here]

                        SUMMARY GUIDELINES:
                        - Highlight key insights and important metrics.
                        - Include actionable items or next steps if present.
                        - Keep it clear, concise, and focused on utility.
                        - Avoid unnecessary details or jargon.

                        STRICT CONTRACT:
                        - Output ONLY the text summary wrapped in the specified tags.
                        """)

        text = await self.llm_model.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        return text
