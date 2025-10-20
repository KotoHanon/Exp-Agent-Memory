import os
import shutil
import sys

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

    working = WorkingMemory()
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
    )

    semantic_fvs = FaissVectorStore()
    print(semantic_fvs.add([new_sem_mem]))
    
    result = semantic_fvs.query(query_text="baseline lacks weather variability")[0][1]

    print(result.detail)
    memory_id = result.id

    semantic_fvs.delete([memory_id])

    renew_sem_mem = semantic.add(
        summary="New one.",
        detail="I am the new one.",
        source_ids=["idea_aug_001"],
        tags=("augmentation", "robustness"),
        confidence=0.7,
    )

    print(semantic_fvs.add([renew_sem_mem]))

    semantic_fvs.save("test")

    new_semantic_fvs = FaissVectorStore()
    new_semantic_fvs.load("test")


    retriever = MemoryRetriever(working, episodic, semantic)
    bundle = retriever.build_context(
        query="fog augmentation robustness",
        idea_id="idea_aug_001",
        tags=("augmentation",),
    )

    print("Working memory snapshot:")
    print(bundle["working_memory"])
    print("\nTop episodic memories:")
    for record in bundle["episodic"]:
        print(f"- {record['summary']} [{record['stage']}]")
    print("\nRelevant semantic insights:")
    for record in bundle["semantic"]:
        print(f"- {record['summary']} (confidence={record['confidence']})")


if __name__ == "__main__":
    main()
