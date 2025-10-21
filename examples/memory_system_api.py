import os
import shutil
import sys
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator, validate_call
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

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
    model_path: str = Field("./.cache/all-MiniLM-L6-v2", description="Path to the model used for vector embeddings.")

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
    def size(self) -> int:
        ...

    @abstractmethod
    def add(self, memories: List[Union[SemanticRecord, EpisodicRecord]]) -> bool:
        ...
    
    @abstractmethod
    def update(self, memories: List[Union[SemanticRecord, EpisodicRecord]]) -> bool:
        ...
    
    @abstractmethod
    def batch_memory_process(self, memories: List[Union[SemanticRecord, EpisodicRecord]]) -> bool:
        ...
    
    @abstractmethod
    def delete(self, mids: List[str]) -> bool:
        ...
    
    @abstractmethod
    def query(self, query_text: str, limit: int = 5, filters: Dict | None = None) -> List[Tuple[float, List[Union[SemanticRecord, EpisodicRecord]]]]:
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

    def size(self) -> int:
        return self.vector_store._get_record_nums()

    def add(self, memories: List[Union[SemanticRecord, EpisodicRecord]] = None) -> bool:
        try:
            self.vector_store.add(memories) # Add new memory to FAISS vectorstore.
            return True
        except Exception as e:
            print(f"Error adding memories: {e}")
            return False
        # TODO: working/procedural memory system 
    
    def update(self, memories: List[SemanticRecord] = None) -> bool:
        try:
            self.vector_store.update(memories) # Update new memory to FAISS vectorstore.
            return True
        except Exception as e:
            print(f"Error updating memories: {e}")
            return False
        # TODO: working/procedural memory system 

    def batch_memory_process(self, memories: List[Union[SemanticRecord, EpisodicRecord]] = None) -> bool:
        '''If you can not distinguish memories need to be add or update, use this method.'''
        try:
            self.vector_store.batch_memory_process(memories)
            return True
        except Exception as e:
            print(f"Error processing memories: {e}")
            return False
    
    def delete(self, mids: List[str]) -> bool:
        try:
            self.vector_store.delete(mids)
            return True
        except Exception as e:
            print(f"Error deleting memories: {e}")
            return False
        return 
    
    def query(self, query_text: str, limit: int = 5, filters: Dict | None = None) -> List[Tuple[float, List[Union[SemanticRecord, EpisodicRecord]]]]:
        results = self.vector_store.query(query_text, limit=limit, filters=filters)
        return results
    
    def save(self, path: str) -> bool:
        try:
            self.vector_store.save(path)
            return True
        except Exception as e:
            return False
    
    def load(self, path: str) -> bool:
        try:
            self.vector_store.load(path)
            return True
        except Exception as e:
            return False

        
