import os
import shutil
import sys
from api.faiss_memory_system_api import FAISSMemorySystem

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from memory_system import FaissVectorStore



def main() -> None:
    log_file = open("ltm_test_output.log", "w")
    sys.stdout = log_file

    print('''--------------------Init test--------------------''')
    semantic_memory_store = FAISSMemorySystem(
        memory_type="semantic",
    )
    sem_rec = semantic_memory_store.instantiate_sem_record(
        summary="Fog augmentations most beneficial when baseline lacks weather variability.",
        detail="Prioritize fog injection when training data is captured in controlled conditions.",
        source_ids=["idea_aug_001"],
        tags=("augmentation", "robustness"),
        confidence=0.7,
    )
    print(f"SemanticRecord instantiation result: {sem_rec.to_dict()}")

    episodic_memory_store = FAISSMemorySystem(
        memory_type="episodic",
    )
    epi_rec = episodic_memory_store.instantiate_epi_record(
        idea_id="idea_aug_001",
        stage="execution",
        summary="Ran experiments with fog augmentations; accuracy +3%.",
        detail={"metrics": {"accuracy": 0.73}, "baseline": 0.7},
        tags=("execution", "augmentation"),
    )
    print(f"EpisodicRecord instantiation result: {epi_rec.to_dict()}")

    procedural_memory_store = FAISSMemorySystem(
        memory_type="procedural",
    )
    proc_rec = procedural_memory_store.instantiate_proc_record(
        name="data_augmentation_procedure",
        description="Procedure to implement data augmentations for model robustness.",
        steps=[
            "Identify target corruptions based on deployment scenarios.",
            "Implement augmentation functions for each corruption type.",
            "Integrate augmentations into data loading pipeline.",
            "Validate effectiveness through controlled experiments.",
        ],
        code="def augment_data(data):\n    # Apply augmentations\n    return augmented_data",
        tags=("procedure", "augmentation"),
    )
    print(f"ProceduralRecord instantiation result: {proc_rec.to_dict()}")

    print('''--------------------Memory add test--------------------''')
    print(f"SemanticRecord adding test: {semantic_memory_store.add([sem_rec])}")
    print(f"EpisodicRecord adding test: {episodic_memory_store.add([epi_rec])}")

    print('''--------------------Memory update test--------------------''')
    sem_rec.update(
        detail="Updated detail: prioritize fog and rain injection.",
        confidence=0.8,
    )
    print(f"SemanticRecord updating test: {semantic_memory_store.update([sem_rec])}")

    print('''--------------------Memory batch-processing test--------------------''')
    new_sem_rec = semantic_memory_store.instantiate_sem_record(
        summary="Rain augmentations also improve robustness in low-light conditions.",
        detail="Incorporate rain effects when training data is captured at night or in dim environments.",
        source_ids=["idea_aug_001"],
        tags=("augmentation", "robustness"),
        confidence=0.6,
    )
    print(f"SemanticRecord batch-processing test: {semantic_memory_store.batch_memory_process([sem_rec, new_sem_rec])}")

    print('''--------------------Memory query(embedding) test--------------------''')
    results = semantic_memory_store.query("augmentations robustness weather", method="embedding", limit=2)
    print(f"SemanticRecord query test results: {[{'score': r[0], 'record': r[1].to_dict()} for r in results]}")

    print('''--------------------Memory query(bm25) test--------------------''')
    results = semantic_memory_store.query("augmentations weather", method="bm25" ,limit=2)
    print(f"SemanticRecord query test results: {[{'score': r[0], 'record': r[1].to_dict()} for r in results]}")

    print('''--------------------Memory query(overlapping) test--------------------''')
    results = semantic_memory_store.query("augmentations weather", method="overlapping" ,limit=2)
    print(f"SemanticRecord query test results: {[{'score': r[0], 'record': r[1].to_dict()} for r in results]}")

    print('''--------------------Memory size test--------------------''')
    print(f"SemanticRecord size test: {semantic_memory_store.size()}")
    print(f"EpisodicRecord size test: {episodic_memory_store.size()}")

    print('''--------------------Memory save test--------------------''')
    print(f"SemanticMemoryStore save test: {semantic_memory_store.save('sem_test')}")
    print(f"EpisodicMemoryStore save test: {episodic_memory_store.save('epi_test')}")

    print('''--------------------Memory load test--------------------''')
    another_semantic_memory_store = FAISSMemorySystem(memory_type="semantic")
    print(f"SemanticMemoryStore load test: {another_semantic_memory_store.load('sem_test')}")
    print(f"New SemanticMemoryStore record nums: {another_semantic_memory_store.size()}")
    another_episodic_memory_store = FAISSMemorySystem(memory_type="episodic")
    print(f"EpisodicMemoryStore load test: {another_episodic_memory_store.load('epi_test')}")
    print(f"New EpisodicMemoryStore record nums: {another_episodic_memory_store.size()}")

    print('''--------------------Memory delete test--------------------''')
    delete_ids = [r[1].id for r in results]
    print(f"SemanticRecord delete test: {another_semantic_memory_store.delete(delete_ids)}")
    print(f"After delete SemanticRecord size: {another_semantic_memory_store.size()}")

    print("--------------------Memory fetch(by ids) test--------------------")
    fetch_ids = [r[1].id for r in results]
    fetched_records = semantic_memory_store.get_records_by_ids(fetch_ids)
    print(f"SemanticRecord fetch test results: {[r.to_dict() for r in fetched_records]}")

    print("--------------------Memory fetch(by last k) test--------------------")
    print(semantic_memory_store.get_last_k_records(2))
    print(semantic_memory_store.get_last_k_records(100))
    print(episodic_memory_store.get_last_k_records(1))
    print(episodic_memory_store.get_last_k_records(100))

    log_file.close()

if __name__ == "__main__":
    main()
