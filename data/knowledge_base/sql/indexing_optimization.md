# SQL Indexing, Execution Plans, and Query Optimization

## Index Types and Internals
- **B-Tree Indexes**: Balanced tree structures storing keys in sorted order with page pointers.
  - Ideal for exact equality matches (`=`), range queries (`<`, `<=`, `>`, `>=`, `BETWEEN`), and prefix `LIKE 'abc%'` queries.
  - Search, insertion, and deletion run in $O(\log N)$ disk page reads.
- **Hash Indexes**: Fast $O(1)$ equality lookup (`=`), but cannot serve range queries or `ORDER BY`.
- **Composite (Multi-column) Indexes**:
  - Index created on `(col_a, col_b, col_c)`.
  - **Leftmost Prefix Rule**: Can satisfy queries filtering on `(col_a)`, `(col_a, col_b)`, or `(col_a, col_b, col_c)`. Cannot serve queries filtering only on `(col_b)` or `(col_c)` without a full index scan.
- **Covering Index**: An index that contains all columns requested by a query (`SELECT` and `WHERE`), eliminating the need for a secondary table heap lookup (Index-Only Scan).

## Query Optimization and Execution Plans
- **`EXPLAIN` / `EXPLAIN ANALYZE`**:
  - `Seq Scan` (Sequential table scan): Reads all table blocks. Appropriate for small tables or when a query fetches $>20-30\%$ of total table rows.
  - `Index Scan`: Traverses B-tree to get heap row pointers (TIDs) and retrieves data from table pages.
  - `Bitmap Index Scan / Bitmap Heap Scan`: Constructs a bitmap of matching pages from the index, sorts physical block numbers to make I/O sequential, and reads heap pages.
  - `Nested Loop Join`: Loops outer rows and queries inner index; fast for small outer sets.
  - `Hash Join`: Builds in-memory hash table on inner dataset, then probes outer rows; ideal for large unindexed joins.
  - `Merge Join`: Joins two pre-sorted datasets; optimal when inputs are already indexed or sorted.

## SARGable (Search Argument Able) Queries
Predicates that allow index tree traversal:
- **Non-SARGable**: `WHERE YEAR(created_at) = 2024` or `WHERE LOWER(email) = 'user@example.com'`. (Applying functions to indexed columns forces full table scan unless an expression index is created).
- **SARGable Equivalent**: `WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01'`.
