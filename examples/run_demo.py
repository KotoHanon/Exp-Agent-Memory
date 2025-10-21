import os
import shutil
import sys
from memory_system_api import FAISSMemorySystem

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from memory_system import (
    EpisodicMemory,
    MemoryRetriever,
    SemanticMemory,
    WorkingMemory,
    FaissVectorStore,
)


def main() -> None:
    data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory_data")
    if os.path.isdir(data_root):
        shutil.rmtree(data_root)

    '''working = WorkingMemory()
    episodic = EpisodicMemory()
    semantic = SemanticMemory()

    working.update_goal("Validate data augmentation idea for model robustness.")
    working.add_hypothesis("Augmentations reduce overfitting on unseen corruption.")
    working.set_plan(
        [
            "Generate synthetic corruptions for validation set.",
            "Compare robustness metrics against baseline.",
            "Summarize findings and update hypothesis.",
        ]
    )
    working.add_active_tool("dataset_manage")

    episodic.add(
        idea_id="idea_aug_001",
        stage="planning",
        summary="Outlined augmentation strategy and selected candidate corruptions.",
        detail={"corruptions": ["fog", "gaussian_noise"], "dataset": "cifar10"},
        tags=("planning", "augmentation"),
    )
    episodic.add(
        idea_id="idea_aug_001",
        stage="execution",
        summary="Ran experiments with fog augmentations; accuracy +3%.",
        detail={"metrics": {"accuracy": 0.73}, "baseline": 0.7},
        tags=("execution", "augmentation"),
    )

    new_sem_mem = semantic.add(
        summary="Fog augmentations most beneficial when baseline lacks weather variability.",
        detail="Prioritize fog injection when training data is captured in controlled conditions.",
        source_ids=["idea_aug_001"],
        tags=("augmentation", "robustness"),
        confidence=0.7,
    )'''

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
    print(semantic_memory_store.add([new_sem_rec]))
    print(f"SemanticRecord batch-processing test: {semantic_memory_store.batch_memory_process([sem_rec, new_sem_rec])}")

    print('''--------------------Memory query test--------------------''')
    results = semantic_memory_store.query("augmentation robustness weather", limit=2)
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


    '''print("Working memory snapshot:")
    print(bundle["working_memory"])
    print("\nTop episodic memories:")
    for record in bundle["episodic"]:
        print(f"- {record['summary']} [{record['stage']}]")
    print("\nRelevant semantic insights:")
    for record in bundle["semantic"]:
        print(f"- {record['summary']} (confidence={record['confidence']})")'''


if __name__ == "__main__":
    main()
