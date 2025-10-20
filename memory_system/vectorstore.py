from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from .models import SemanticRecord
from .utils import _nomralize_embedding, _jsonable_meta
from sentence_transformers import SentenceTransformer

import numpy as np
import json, os
import faiss

base_dir = os.path.dirname(os.path.abspath(__file__))

class VectorStore(ABC):
    @abstractmethod
    def add(self, raws) -> List[int]:
        ...
    
    @abstractmethod
    def update(self, raws) -> int:
        ...

    @abstractmethod
    def query(self, query_text: str, limit: int = 5, filters: Dict | None = None) -> List[Tuple[float, Dict]]:
        ...

    @abstractmethod
    def delete(self, mids: List[str]) -> int:
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...

class FaissVectorStore(VectorStore):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(os.path.join(base_dir, "./.cache/all-MiniLM-L6-v2"))
        self.index = None
        self.dim = None
        self.meta: Dict[int, Dict] = {} # {id: SemanticRecord}
        self.fidmap2mid: Dict[int, string] = {} #{faiss_id: memory_id}
        self._next_id = 0
    
    def _embed(self, texts: list[str]):
        # Normalize for FAISS store and query.
        embs = _nomralize_embedding(self.model.encode(texts))
        return np.array(embs, dtype="float32")
    
    def _ensure_index(self, dim: int):
        if self.index is None:
            base = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIDMap2(base)
            self.dim = dim
    
    def _separate_add_and_update(self, raws: List[SemanticRecord]) -> Tuple[List[SemanticRecord], List[SemanticRecord]]:
        reverse_map = {mid : fid for fid, mid in self.fidmap2mid.items()}
        adds, updates = [], []
        for raw in raws:
            if raw.id in reverse_map:
                updates.append(raw)
            else:
                adds.append(raw)
        return adds, updates
    
    def add(self, raws: List[SemanticRecord]) -> List[int]:
        if len(raws) == 0:
            return []
        
        # check existing ids
        texts = [raw.detail for raw in raws]
        mids = [raw.id for raw in raws]
        ids = np.arange(self._next_id, self._next_id + len(raws), dtype="int64")
        self.fidmap2mid.update({int(fid) : mid for fid, mid in zip(ids, mids)})
        self._next_id += len(raws)
        vecs = self._embed(texts) # [N, dim]
        self._ensure_index(vecs.shape[1])
        # write for FAISS
        self.index.add_with_ids(vecs, ids)

        for i, r in zip(ids, raws):
            # bind data for every id
            self.meta[int(i)] = r
        return ids.tolist()

    def update(self, raws: List[SemanticRecord]) -> List[int]:
        if len(raws) == 0:
            return []

        ids = [fid for fid, mid in self.fidmap2mid.items()]
        assert len(ids) == len(raws), "The number of records to update must match the number of existing records."

        for i, r in zip(ids, raws):
            # bind data for every id
            self.meta[int(i)] = r
        return ids.tolist()

    def batch_memory_process(self, raws: List[SemanticRecord]) -> None:
        adds, updates = self._separate_add_and_update(raws)
        self.add(adds)
        self.update(updates)

    def query(self, query_text: str, limit: int = 5, filters: Dict | None = None) -> List[Tuple[float, Dict]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q = self._embed([query_text])
        D, I = self.index.search(q, limit)
        results = []
        for score, _id in zip(D[0], I[0]):
            if _id == -1:
                continue
            md = self.meta.get(int(_id), {})
            if filters:
                ok = all(md.get(k2) == v2 for k2, v2 in filters.items())
                if not ok:
                    continue
            results.append((float(score), md))
        return results
    
    def delete(self, mids: List[str]) -> None:
        if self.index is None or not mids:
            return        
        ids = [fid for fid, mid in self.fidmap2mid.items() if mid in mids]
        indices = np.ascontiguousarray(ids, dtype="int64")
        try:
            sel = faiss.IDSelectorBatch(indices)
        except TypeError:
            sel = faiss.IDSelectorBatch(len(indices), faiss.swig_ptr(indices)) 

        removed = int(self.index.remove_ids(sel))

        for i in ids:
            self.meta.pop(int(i), None)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"meta": _jsonable_meta(self.meta), "next_id": self._next_id}, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        self.meta = {int(k) : v for k, v in data["meta"].items()}
        self._next_id = int(data.get("faiss_id", self.index.ntotal))
        self.dim = self.index.d

            
