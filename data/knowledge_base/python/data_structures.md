# Python Data Structures: Internals and Complexities

## Overview
Python provides built-in mutable and immutable data structures. Understanding their underlying memory layouts, time complexities, and hash table implementations is essential for engineering performant systems.

## Lists vs. Tuples
- **Lists (`list`)**:
  - Mutable, dynamic arrays implemented under CPython as contiguous arrays of pointers (`PyObject**`).
  - Amortized $O(1)$ append time due to over-allocation growth pattern $(0, 4, 8, 16, 24, 32, ...)$.
  - Slicing creates a shallow copy in $O(k)$ time.
  - Insert and delete operations at arbitrary indices take $O(n)$ time because elements must be shifted in memory.
- **Tuples (`tuple`)**:
  - Immutable sequences. Once created, length and elements cannot be modified.
  - Memory optimization: Tuples of small sizes are cached by CPython via free lists (`tuple_free_list`), avoiding heap reallocation.
  - Tuple size in bytes is strictly smaller than a list of equivalent length because it does not allocate extra capacity for dynamic resizing.

## Dictionaries (`dict`)
- High-performance key-value mapping built on open addressing hash tables.
- Since Python 3.6+, dictionaries preserve insertion order using a compact array representation:
  - An `indices` sparse array containing table indices.
  - An `entries` dense array containing `[hash, key_ptr, value_ptr]` in order of insertion.
- **Lookup, Insertion, and Deletion**:
  - Average Case: $O(1)$
  - Worst Case: $O(n)$ under catastrophic hash collisions.
- **Hash Collisions**: Handled via perturb-based open addressing: `j = ((5*j) + 1 + perturb) & mask`.

## Sets (`set`)
- Mutable, unordered collection of unique hashable elements.
- Implemented similarly to `dict` keys without value pointers.
- Set operations:
  - Membership testing (`in`): Average $O(1)$, Worst $O(n)$.
  - Union ($A \cup B$): $O(|A| + |B|)$.
  - Intersection ($A \cap B$): $O(\min(|A|, |B|))$.
  - Difference ($A \setminus B$): $O(|A|)$.

## Hashing and the `__hash__` Contract
An object is hashable if it has a hash value that never changes during its lifetime and can be compared to other objects (`__eq__`).
Mutable containers (like `list` or `dict`) do not implement `__hash__` to prevent mutating a key after insertion into a hash table, which would break bucket addressing.
