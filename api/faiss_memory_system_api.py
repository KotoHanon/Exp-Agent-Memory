import os
import shutil
import sys

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator, validate_call
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union
from memory_system import (
    FaissVectorStore,
    SemanticRecord,
    EpisodicRecord,
    ProceduralRecord,
    OpenAIClient,
)
from memory_system.utils import now_iso, new_id, _transfer_dict_to_semantic_text
from memory_system.denstream import DenStream
from .base_memory_system_api import MemorySystem, MemorySystemConfig, MemoryRecordPayload
from collections import defaultdicts

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class FAISSMemorySystem(MemorySystem):
    def __init__(self, **kwargs):
        cfg = MemorySystemConfig(**kwargs)

        self.memory_type = cfg.memory_type
        self.vector_store = FaissVectorStore(cfg.model_path)
        self.llm = OpenAIClient(model_name=cfg.llm_name)

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
    
    def instantiate_epi_record(self, eps: float = 0.6, beta: float = 0.5, mu: float = 4, **kwargs) -> EpisodicRecord:
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
        record.embedding = self.vector_store._embed(_transfer_dict_to_semantic_text(record.detail))
        self.cluster_machine = DenStream(eps=eps, beta=beta, mu=mu)        
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
    
    def update(self, memories: List[Union[SemanticRecord, ProceduralRecord]] = None) -> bool:
        try:
            self.vector_store.update(memories) # Update new memory to FAISS vectorstore.
            return True
        except Exception as e:
            print(f"Error updating memories: {e}")
            return False

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
            if self.memory_type == "episodic":
                self.cluster_machine = DenStream() # Reset clustering machine
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

    async def abstract_episodic_records(
            self, 
            epi_records: List[EpisodicRecord], 
            weight_threshold: float = 5.0, 
            consistency_threshold: float = 0.8) -> List[SemanticRecord]:
        assert self.memory_type == "episodic", "Clustering is only supported for episodic memory type."
        cidmap2mid: Dict[int, List] = defaultdict(list) # {cluster_id: semantic_record_id}
        result: Dict[EpisodicRecord, Dict] = {} # {EpisodicRecord: info}
        for epi in epi_records:
            info = denstream.process(point=epi.embedding, now=epi.created_at)
            cidmap2mid[info['cluster_id']].append(epi.id)
            result.update({epi: info})
        
        score = dict(sorted(
            self.cluster_machine.cidmap2cluster.items(),
            key=lambda item: item[1].avg_pairwise_cos(), 
            reverse=True
        ))

        for cl in score.values():
            if cl.cluster_weight >= weight_threshold and cl.avg_pairwise_cos() >= consistency_threshold:
                # TODO: Generate semantic record for this cluster
                sem_record_dict = self.llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)

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

        
