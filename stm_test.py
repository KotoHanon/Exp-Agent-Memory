import os
import shutil
import sys

from api.slot_process_api import SlotProcess
from memory_system import WorkingSlot

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def main() -> None:
    log_file = open("stm_test_output.log", "w")
    sys.stdout = log_file

    print('''--------------------Init test--------------------''')
    working_slot = WorkingSlot(
        stage="analysis",
        topic="science",
        summary="This is a test summary.",
        attachments={
            "key1": {"value1": 0.6},
            "key2": {"value2": 0.8},
            "metrics": {"accuracy": 0.95, "loss": 0.05},
            "result": {"filter": True},
        },
        tags=["test", "stm"]
    )
    print(f"WorkingSlot initialized: {working_slot.to_dict()}")

    print('''------------------SlotProcess test: add------------------''')
    slot_process = SlotProcess()
    slot_process.add_slot(working_slot)
    print(f"Slot queue size: {slot_process.get_queue_size()}")

    print('''--------------------SlotProcess test: clear--------------------''')
    slot_process.clear_queue()
    print(f"Slot queue size after clear: {slot_process.get_queue_size()}")

    print('''--------------------SlotProcess test: fliter and route--------------------''')
    slot_process.add_slot(working_slot)
    result = slot_process.filter_and_route_slots()
    print(f"Filtered and routed slots: {len(result)}")
    print(f"Memory type: {result[0].get('memory_type')}")

    log_file.close()

if __name__ == "__main__":
    main()