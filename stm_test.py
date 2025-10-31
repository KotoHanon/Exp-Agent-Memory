import os
import shutil
import sys

from api.slot_process_api import SlotProcess
from memory_system import WorkingSlot
from textwrap import dedent

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def main() -> None:
    log_file = open("stm_test_output.log", "w")
    sys.stdout = log_file

    print('''--------------------Init test--------------------''')
    trivial_working_slot = WorkingSlot(
        stage="analysis",
        topic="science",
        summary="This is a test summary.",
        attachments={
            "key1": {"value1": 0.6},
            "key2": {"value2": 0.8},
            "metrics": {"accuracy": 0.95, "loss": 0.05},
        },
        tags=["test", "stm"]
    )
    valuable_working_slot = WorkingSlot(
        stage="reflection",
        topic="reinforcement_learning",
        summary="Analyzed the performance impact of integrating episodic memory into the RL agent; found significant stability improvements during long-horizon exploration.",
        attachments={
            "experiment": {
                "env": "AntMaze-Medium",
                "agent": "Dynamic LEGOMem",
                "baseline": "MBPO",
                "config": {"seed": 42, "steps": 5e5}
            },
            "metrics": {
                "return_mean": 0.87,
                "return_std": 0.03,
                "success_rate": 0.82,
                "stability_index": 0.91
            },
            "insight": {
                "observation": "Episodic recall reduced catastrophic forgetting in transition prediction.",
                "hypothesis": "Stabilization emerges from periodic semantic consolidation.",
                "next_step": "Integrate active retrieval routing into the policy update loop."
            },
        },
        tags=["analysis", "rl", "stability"]
    )

    print(f"Trivial WorkingSlot initialized: {trivial_working_slot.to_dict()}")
    print(f"Valuable WorkingSlot initialized: {valuable_working_slot.to_dict()}")

    print('''------------------SlotProcess test: add------------------''')
    slot_process = SlotProcess()
    slot_process.add_slot(trivial_working_slot)
    slot_process.add_slot(valuable_working_slot)
    print(f"Slot queue size: {slot_process.get_queue_size()}")

    print('''--------------------SlotProcess test: clear--------------------''')
    slot_process.clear_queue()
    print(f"Slot queue size after clear: {slot_process.get_queue_size()}")

    print('''--------------------SlotProcess test: fliter and route--------------------''')
    slot_process.add_slot(trivial_working_slot)
    slot_process.add_slot(valuable_working_slot)
    result = slot_process.filter_and_route_slots()
    print(f"Filtered and routed slots: {len(result)}")
    if len(result) > 0:
        print(f"Memory type: {result[0].get('memory_type')}")

        print('''--------------------SlotProcess test: transfer--------------------''')
        for r in result:
            print(dedent(f"""
            
            Transfer result:

            {slot_process.transfer_slot_to_text(r.get('slot'))}
            """))


    log_file.close()

if __name__ == "__main__":
    main()