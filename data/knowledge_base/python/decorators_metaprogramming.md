# Python Decorators and Metaprogramming

## Closures and First-Class Functions
In Python, functions are first-class citizens: they can be passed as arguments, assigned to variables, returned from other functions, and stored in data structures.
A **closure** occurs when an inner function retains access to variables in its enclosing lexical scope even after the outer function has finished executing.

## Function Decorators
A decorator is a callable that takes a function as input and returns a modified or wrapped function. The `@decorator` syntax is syntactic sugar:
```python
@my_decorator
def calculate(x):
    return x * 2

# Equivalent to:
calculate = my_decorator(calculate)
```

## Preserving Metadata with `functools.wraps`
When wrapping a function, the wrapper replaces the original function's `__name__`, `__doc__`, and `__module__`. Using `@functools.wraps(fn)` copies these attributes back to the wrapper, preserving introspection and debugging integrity.

## Parameterized Decorators
To pass arguments to a decorator, an extra level of nesting is required (a decorator factory):
```python
def retry(attempts=3, delay=1.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == attempts - 1:
                        raise e
                    time.sleep(delay)
        return wrapper
    return decorator
```

## Descriptors Protocol
A descriptor is an object attribute with "binding behavior" whose attribute access is overridden by methods in the descriptor protocol:
- `__get__(self, obj, type=None)`
- `__set__(self, obj, value)`
- `__delete__(self, obj)`
If an object defines `__set__` or `__delete__`, it is a **data descriptor**. If it only defines `__get__`, it is a **non-data descriptor** (like standard methods).
The `@property` decorator is implemented as a data descriptor.
