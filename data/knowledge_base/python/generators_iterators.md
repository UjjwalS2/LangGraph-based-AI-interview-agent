# Python Iterators and Generators

## The Iterator Protocol
An iterator in Python is an object that implements the iterator protocol, consisting of two methods:
1. `__iter__()`: Returns the iterator object itself.
2. `__next__()`: Returns the next value from the sequence. When no items remain, it raises the `StopIteration` exception.

An **iterable** is any object that defines `__iter__()` (returning an iterator) or `__getitem__()` with sequential integer indices starting at 0.

## Generators and the `yield` Statement
Generators are functions that produce a sequence of results lazily instead of computing them upfront in memory:
- When a generator function is called, it returns a generator iterator object without executing the function body immediately.
- Calling `next(gen)` starts or resumes execution until a `yield` expression is encountered.
- Execution pauses, and the local execution state, variable bindings, and instruction pointer are preserved in the generator frame.
- Successive calls resume directly after the previous `yield`.

## Memory Advantages of Generators
- **Lazy Evaluation**: Elements are generated on-the-fly ($O(1)$ memory consumption), making it possible to stream infinite sequences or multi-gigabyte log files without exhausting RAM.
- **Generator Expressions**: Concise syntax `(x**2 for x in iterable)` returning an iterator object with minimal memory footprint compared to list comprehensions `[x**2 for x in iterable]`.

## Delegation with `yield from`
- `yield from iterable` delegates value generation directly to a sub-generator or iterable.
- It automatically establishes a bidirectional communication channel:
  - Propagates `.send(val)`, `.throw(exc)`, and `.close()` directly to the sub-generator.
  - Transparently captures the return value of the sub-generator via `res = yield from subgen()`.
