"""
State checkpointer and persistence provider for LangGraph.
"""

from langgraph.checkpoint.memory import MemorySaver

_GLOBAL_SAVER = None


def get_memory_checkpointer() -> MemorySaver:
    global _GLOBAL_SAVER
    if _GLOBAL_SAVER is None:
        _GLOBAL_SAVER = MemorySaver()
    return _GLOBAL_SAVER
