import os
import shutil
import sys
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator, validate_call
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from memory_system import (
    EpisodicMemory,
    SemanticMemory,
    FaissVectorStore,
    SemanticRecord,
    EpisodicRecord,
    ProceduralRecord,
)
from memory_system.utils import now_iso, new_id
from .base_memory_system_api import MemorySystem, MemorySystemConfig, MemoryRecordPayload

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
            tags=cfg.tags,
            created_at=now_iso(),
        )        
        return record

    def instantiate_proc_record(self, **kwargs) -> ProceduralRecord:
        cfg = MemoryRecordPayload(**kwargs)
        record = ProceduralRecord(
            id=new_id("proc"),
            name=cfg.name,
            description=cfg.description,
            steps=cfg.steps,
            code=cfg.code,
            tags=cfg.tags,
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        return record

    def size(self) -> int:
        return self.vector_store._get_record_nums()

    def get_records_by_ids(self, mids: List[str]) -> Union[List[SemanticRecord], List[EpisodicRecord], List[ProceduralRecord]]:
        reverse_map = {mid: fid for fid, mid in self.vector_store.fidmap2mid.items()}
        records = []
        for mid in mids:
            fid = reverse_map.get(mid, None)
            try:
                record = self.vector_store.meta[fid]
            except KeyError as e:
                print(f"Record with id {mid} not found: {e}")
                continue
            records.append(record)
        return records
    
    def get_last_k_records(self, k: int) -> Tuple[Union[List[SemanticRecord], List[EpisodicRecord], List[ProceduralRecord]], int]:
        if k >= self.size():
            return [record for record in self.vector_store.meta.values()], self.size()
        
        else:
            sorted_fids = sorted(self.vector_store.fidmap2mid.keys(), reverse=True)
            return [self.vector_store.meta[fid] for fid in sorted_fids[:real_k]], k
        
    def is_exists(self, mids: List[str]) -> List[bool]:
        reverse_map = {mid: fid for fid, mid in self.vector_store.fidmap2mid.items()}
        results = []
        for mid in mids:
            fid = reverse_map.get(mid, None)
            if fid is not None and fid in self.vector_store.meta:
                results.append(True)
            else:
                results.append(False)
        return results
        
    def add(self, memories: List[Union[SemanticRecord, EpisodicRecord, ProceduralRecord]] = None) -> bool:
        try:
            self.vector_store.add(memories) # Add new memory to FAISS vectorstore.
            return True
        except Exception as e:
            print(f"Error adding memories: {e}")
            return False
        # TODO: working memory system 
    
    def update(self, memories: List[Union[SemanticRecord, ProceduralRecord]] = None) -> bool:
        try:
            self.vector_store.update(memories) # Update new memory to FAISS vectorstore.
            return True
        except Exception as e:
            print(f"Error updating memories: {e}")
            return False
        # TODO: working memory system 

    def batch_memory_process(self, memories: List[Union[SemanticRecord, EpisodicRecord, ProceduralRecord]] = None) -> bool:
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
    
    def query(self, query_text: str, method: str = "embedding", limit: int = 5, filters: Dict | None = None) -> List[Tuple[float, List[Union[SemanticRecord, EpisodicRecord, ProceduralRecord]]]]:
        try:
            results = self.vector_store.query(query_text, method=method, limit=limit, filters=filters)
        except Exception as e:
            print(f"Error querying memories: {e}")
            results = []
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

        
