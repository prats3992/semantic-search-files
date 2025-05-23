# Python Data Structures

Python offers a rich set of built-in data structures, each with its own characteristics and use cases.

Common data structures in Python include:
- **Lists**: Ordered, mutable sequences. Represented by `[]`.
  - Can store items of different types.
  - Support indexing, slicing, appending, inserting, and removing elements.
  - Useful for ordered collections where elements might change, be added, or removed.
  - Example: `my_list = [1, "hello", 3.14]`
- **Tuples**: Ordered, immutable sequences. Represented by `()`.
  - Once created, their elements cannot be changed, added, or removed.
  - Often used for fixed collections of items, like coordinates `(x, y)` or RGB color values.
  - Can be used as keys in dictionaries if all their elements are immutable.
  - Generally more memory-efficient than lists for the same data.
  - Example: `my_tuple = (1, "world", True)`
- **Dictionaries**: Unordered (in Python < 3.7, ordered in CPython 3.6+ and officially in Python 3.7+) collections of key-value pairs. Represented by `{}`.
  - Keys must be unique and immutable (e.g., strings, numbers, tuples).
  - Values can be of any type and can be duplicated.
  - Highly optimized for retrieving values when the key is known.
  - Useful for storing data associated with unique identifiers.
  - Example: `my_dict = {"name": "Alice", "age": 30}`
- **Sets**: Unordered collections of unique, immutable elements. Represented by `{}` (but an empty set is `set()`).
  - Automatically handle uniqueness; duplicate elements are ignored.
  - Support mathematical set operations like union, intersection, difference, and symmetric difference.
  - Useful for membership testing, removing duplicates from a sequence, and performing set-based logic.
  - Example: `my_set = {1, 2, 3, 2, 1}` results in `{1, 2, 3}`

Other notable structures:
- **Strings**: Ordered, immutable sequences of characters. While often treated as a basic type, they are a sequence type.
- **collections module**: Provides more specialized data structures:
  - `collections.deque`: A list-like container with fast appends and pops on either end.
  - `collections.Counter`: A dict subclass for counting hashable objects.
  - `collections.OrderedDict`: A dict subclass that remembers the order entries were added (less critical since Python 3.7+ dicts are ordered).
  - `collections.defaultdict`: A dict subclass that calls a factory function to supply missing values.
  - `collections.namedtuple()`: Factory function for creating tuple subclasses with named fields.

Choosing the right data structure is crucial for writing efficient and readable Python code. The choice depends on the specific requirements of the task, such as whether the order of elements matters, whether elements need to be unique, or whether the collection needs to be mutable.
