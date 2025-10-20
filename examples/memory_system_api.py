import os
import shutil
import sys
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator, validate_call

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from memory_system import (
    EpisodicMemory,
    MemoryRetriever,
    SemanticMemory,
    WorkingMemory,
    FaissVectorStore,
    SemanticRecord,
    EpisodicRecord,
)
from memory_system.utils import now_iso, new_id

class MemorySystemConfig(BaseModel):
    memory_type: Literal["semantic", "episodic", "working"] = "semantic"
    model_path: Optional[str] = Field(None, description="Path to the model used for vector embeddings.")
    memory_store_path: Optional[str] = Field(None, description="Path to the memory store.")

class MemoryRecordPayload(BaseModel):
    summary: str = Field(..., description="A brief summary of the memory.")
    detail: Union[str, dict] = Field(..., description="Detailed information about the memory.")
    stage: str = Field("", description="Stage of the memory.")
    idea_id: str = Field("", description="Unique identifier for the idea.")
    source_ids: Optional[List[str]] = Field(None, description="List of source IDs related to the memory.")
    tags: Optional[Iterable[str]] = Field(None, description="Tags associated with the memory.")
    metrics: Optional[dict] = Field(None, description="Metrics used for experiment.")
    confidence: float = Field(0.0, description="Confidence score of the memory.")

class MemorySystem(ABC):
    @abstractmethod
    def instantiate_sem_record(self, **kwargs) -> SemanticRecord:
        ...
    
    @abstractmethod
    def instantiate_epi_record(self, **kwargs) -> EpisodicRecord:
        ...

    @abstractmethod
    def add(self, List[Union[SemanticRecord, EpisodicRecord]]) -> List[int]:
        ...
    
    @abstractmethod
    def update(self, List[Union[SemanticRecord, EpisodicRecord]]) -> List[int]:
        ...
    
    @abstractmethod
    def batch_memory_process(self, List[Union[SemanticRecord, EpisodicRecord]]) -> List[int]:
        ...
    
    @abstractmethod
    def delete(self, mids: List[str]) -> List[int]:
        ...
    
    @abstractmethod
    def query(self, query_text: str, limit: int = 5, filters: Dict | None = None) -> List[Tuple[float, Dict]]:
        ...
    
    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...

class FAISSMemorySystem(MemorySystem):
    def __init__(self, **kwargs):
        cfg = MemorySystemConfig(**kwargs)

        self.memory_type = cfg.memory_type
        self.vector_store = FaissVectorStore(cfg.model_path)

        if self.memory_type == "semantic":
            self.memory = EpisodicMemory(cfg.memory_store_path)
        elif self.memory_type == "episodic":
            self.memory = SemanticMemory(cfg.memory_store_path)
        elif self.memory_type == "working":
            self.memory = WorkingMemory()
        else:
            raise ValueError(f"Unknown memory type: {self.memory_type}")

    def instantiate_sem_record(self, **kwargs) -> SemanticRecord:
        cfg = MemoryRecordPayload(**kwargs)
        record = SemanticRecord(
            id=new_id("sem"),
            summary=cfg.summary,
            detail=cfg.detail,
            source_ids=cfg.source_ids,
            tags=cfg.tags,
            confidence=cfg.confidence,
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        return record
    
    def instantiate_epi_record(self, **kwargs) -> EpisodicRecord:
        cfg = MemoryRecordPayload(**kwargs)
        record = EpisodicRecord(
            id=new_id("epi"),
            idea_id=cfg.idea_id,
            stage=cfg.stage,
            summary=cfg.summary,
            detail=cfg.detail,
            metrics=cfg.metrics,
            tags=cfg.tags,
            created_at=now_iso(),
        )        
        return record

    def add(self, new_memories: List[Union[SemanticRecord, EpisodicRecord]] = None) -> None:
        self.vector_store.add(new_memorys) # Add new memory to FAISS vectorstore.
        # TODO: working/procedural memory system 
    
    def update(self, updated_memories: List[Union[SemanticRecord, EpisodicRecord]]):
        self.vector_store.update(updated_memories) # Update new memory to FAISS vectorstore.
        # TODO: working/procedural memory system 

    def batch_memory_process(self, memories: List[Union[SemanticRecord, EpisodicRecord]]):
        '''If you can not distinguish memories need to be add or update, use this method.'''
        self.vector_store.batch_memory_process(memories)
    
    def delete(self, mids: List[str]) -> List[int]:
        removed = self.vector_store.delete(mids)
        return removed
    
    def query(self, query_text: str, limit: int = 5, filters: Dict | None = None) -> List[Tuple[float, Dict]]:
        results = self.vector_store.query(query_text, limit=limit, filters=filters)
        return results
    
    def save(self, path: str) -> None:
        self.vector_store.save(path)
    
    def load(self, path: str) -> None:
        self.vector_store.load(path)

        
