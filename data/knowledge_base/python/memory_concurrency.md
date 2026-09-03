# Python Memory Management and Concurrency Models

## CPython Memory Architecture
Memory management in CPython is layered:
1. **PyObject Allocator / PyMalloc**: Allocates small objects ($\le 512$ bytes) using Arenas (256 KB), Pools (4 KB), and Blocks (aligned sizes up to 512 bytes) to minimize OS malloc overhead and heap fragmentation.
2. **Raw Memory Allocator**: Uses system `malloc`/`free` for allocations larger than 512 bytes.

## Reference Counting and Cyclic Garbage Collection
- **Reference Counting**: The primary deallocation mechanism. Every `PyObject` contains an `ob_refcnt` field.
  - Increment on assignment, argument passing, or list insertion.
  - Decrement when variable goes out of scope or is deleted via `del`.
  - When `ob_refcnt == 0`, memory is immediately freed.
- **Cyclic Garbage Collector (`gc`)**:
  - Handles circular references (e.g. object A references B, and B references A) where reference counts never drop to 0.
  - Employs a generational approach with 3 generations (Gen 0, Gen 1, Gen 2).
  - Gen 0 is collected frequently; survivors are promoted to older generations.
  - Uses double-linked lists of trackable container objects and detects unreachable cycles by finding isolated subgraphs.

## The Global Interpreter Lock (GIL)
- A mutex lock preventing multiple native OS threads from executing CPython bytecode simultaneously.
- Protects CPython's reference counts and internal C data structures from race conditions.
- **Impact**:
  - CPU-bound multithreaded tasks do not achieve parallel execution on multi-core processors.
  - I/O-bound tasks release the GIL during blocking system calls (socket read/write, file I/O, sleep), allowing other threads to run concurrently.

## Concurrency Paradigms in Python
1. **`threading`**: Native OS threads with shared memory. Ideal for I/O-bound tasks with network or disk latency. Limited for CPU-heavy computation by the GIL.
2. **`multiprocessing`**: Creates separate OS processes, each with its own Python interpreter, memory space, and GIL. Bypasses GIL limitations for parallel CPU-bound compute; requires IPC (Inter-Process Communication) such as `Pipes`, `Queues`, or shared memory (`multiprocessing.shared_memory`).
3. **`asyncio`**: Single-threaded, cooperative multitasking using an event loop, coroutines (`async def`), and non-blocking I/O (`await`). Extreme concurrency for high-volume network applications without OS thread context-switching overhead.
