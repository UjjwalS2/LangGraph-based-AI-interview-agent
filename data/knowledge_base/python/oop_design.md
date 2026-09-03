# Python Object-Oriented Programming and Method Resolution

## Classes, Inheritance, and Polymorphism
Python supports dynamic object-oriented programming with multiple inheritance, duck typing, and class-level metaprogramming.

## Object Instantiation: `__new__` vs. `__init__`
- `__new__(cls, *args, **kwargs)`: Static method responsible for allocating memory and returning a new instance of class `cls`. Used for subclassing immutable types (like `tuple` or `str`) or implementing the Singleton pattern.
- `__init__(self, *args, **kwargs)`: Instance method responsible for initializing the newly created instance attributes. Does not return any value.

## Multiple Inheritance and C3 Linearization (MRO)
Python uses the C3 Linearization algorithm to determine the Method Resolution Order (MRO):
- Resolves methods deterministically in multiple inheritance hierarchies.
- Guarantees:
  1. Children precede their parents.
  2. The order of parent classes in the class declaration list is preserved (monotonicity).
  3. No duplicate class lookups.
- Inspectable at runtime via `ClassName.__mro__` or `ClassName.mro()`.
- `super()` follows the MRO of the calling instance (`self`), not necessarily the immediate lexical parent in the class definition.

## Abstract Base Classes (ABCs)
The `abc` module provides the `ABC` class and `@abstractmethod` decorator:
- Classes containing abstract methods cannot be instantiated.
- Subclasses must implement all abstract methods before instantiation.
- Supports virtual subclasses via `ABCMeta.register()`, enabling `isinstance()` checks without requiring explicit inheritance.
