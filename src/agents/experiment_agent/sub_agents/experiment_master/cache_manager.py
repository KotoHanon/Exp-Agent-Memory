"""
Cache Manager for Experiment Workflow.

This module provides caching functionality to save and load agent outputs.
"""

import json
import hashlib
import os
from datetime import datetime
from typing import Any, Optional, Dict
from pathlib import Path


class CacheManager:
    """Manages caching of agent outputs during workflow execution."""

    def __init__(self, cache_dir: str = "./cached"):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each agent type
        self.agent_dirs = {
            "pre_analysis": self.cache_dir / "pre_analysis",
            "code_plan": self.cache_dir / "code_plan",
            "code_implement": self.cache_dir / "code_implement",
            "code_judge": self.cache_dir / "code_judge",
            "experiment_execute": self.cache_dir / "experiment_execute",
            "experiment_analysis": self.cache_dir / "experiment_analysis",
        }
        
        for agent_dir in self.agent_dirs.values():
            agent_dir.mkdir(parents=True, exist_ok=True)

    def _generate_cache_key(self, agent_name: str, input_data: str) -> str:
        """Generate a unique cache key based on agent name and input."""
        input_hash = hashlib.sha256(input_data.encode("utf-8")).hexdigest()[:16]
        return input_hash

    def _get_cache_path(self, agent_name: str, cache_key: str) -> Path:
        """Get the file path for a cache entry."""
        agent_dir = self.agent_dirs.get(agent_name, self.cache_dir / agent_name)
        agent_dir.mkdir(parents=True, exist_ok=True)
        return agent_dir / f"{cache_key}.json"

    def _serialize_output(self, output: Any) -> Any:
        """Serialize agent output to JSON-compatible format."""
        if output is None:
            return None
            
        # Handle Pydantic models
        if hasattr(output, "model_dump"):
            return output.model_dump()
        elif hasattr(output, "dict"):
            return output.dict()
        # Handle dataclasses
        elif hasattr(output, "__dataclass_fields__"):
            from dataclasses import asdict
            return asdict(output)
        # Handle dict
        elif isinstance(output, dict):
            return output
        # Handle list
        elif isinstance(output, list):
            return [self._serialize_output(item) for item in output]
        # Handle primitive types
        elif isinstance(output, (str, int, float, bool)):
            return output
        # Fallback: convert to string
        else:
            return str(output)

    def save_cache(
        self,
        agent_name: str,
        input_data: str,
        output: Any,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Save agent output to cache.
        
        Args:
            agent_name: Name of the agent
            input_data: Input string to the agent
            output: Agent output to cache
            metadata: Optional metadata to store
            
        Returns:
            Cache key used for this entry
        """
        cache_key = self._generate_cache_key(agent_name, input_data)
        cache_path = self._get_cache_path(agent_name, cache_key)

        # Serialize output
        serialized_output = self._serialize_output(output)

        # Prepare cache entry
        cache_entry = {
            "agent_name": agent_name,
            "cache_key": cache_key,
            "timestamp": datetime.now().isoformat(),
            "input_summary": input_data[:500] + "..." if len(input_data) > 500 else input_data,
            "output": serialized_output,
            "metadata": metadata or {},
        }

        # Save to file
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_entry, f, indent=2, ensure_ascii=False)

        print(f"[CACHE] Saved {agent_name} output to cache (key: {cache_key})")
        return cache_key

    def load_cache(self, agent_name: str, input_data: str) -> Optional[Any]:
        """Load agent output from cache if available."""
        cache_key = self._generate_cache_key(agent_name, input_data)
        cache_path = self._get_cache_path(agent_name, cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_entry = json.load(f)

            output = cache_entry.get("output")
            timestamp = cache_entry.get("timestamp")
            
            print(f"[CACHE] Loaded {agent_name} output from cache (saved: {timestamp})")
            return output

        except Exception as e:
            print(f"[CACHE] Error loading cache for {agent_name}: {str(e)}")
            return None

    def has_cache(self, agent_name: str, input_data: str) -> bool:
        """Check if cache exists for given agent and input."""
        cache_key = self._generate_cache_key(agent_name, input_data)
        cache_path = self._get_cache_path(agent_name, cache_key)
        return cache_path.exists()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cached entries."""
        stats = {}
        for agent_name, agent_dir in self.agent_dirs.items():
            if agent_dir.exists():
                count = len(list(agent_dir.glob("*.json")))
                stats[agent_name] = count
        return stats

