# SQL Window Functions and Frame Specifications

## Window Functions Overview
Window functions perform calculations across a set of table rows related to the current row without collapsing rows into a single summary output (unlike `GROUP BY`).
Syntax:
```sql
FUNCTION(...) OVER (
    [PARTITION BY partition_col]
    [ORDER BY sort_col [ASC|DESC]]
    [ROWS|RANGE frame_specification]
)
```

## Ranking Functions
1. **`ROW_NUMBER()`**: Assigns a unique sequential integer starting at 1 to each row within the partition, regardless of duplicate sort values.
2. **`RANK()`**: Assigns ranks with gaps for ties. If two rows share rank 1, the subsequent row receives rank 3.
3. **`DENSE_RANK()`**: Assigns consecutive ranks without gaps for ties. If two rows share rank 1, the subsequent row receives rank 2.
4. **`NTILE(n)`**: Divides rows in each partition into $n$ approximately equal buckets and assigns bucket index $(1 \dots n)$.

## Value & Navigation Functions
- **`LEAD(col, offset, default)`**: Returns value from $offset$ rows ahead in the partition.
- **`LAG(col, offset, default)`**: Returns value from $offset$ rows behind in the partition.
- **`FIRST_VALUE(col)`** and **`LAST_VALUE(col)`**: Return the first or last value in the current window frame.

## Window Frame Specifications (`ROWS` vs. `RANGE`)
- **`ROWS`**: Operates on physical row counts (e.g. `ROWS BETWEEN 2 PRECEDING AND CURRENT ROW`).
- **`RANGE`**: Operates on logical value offsets matching the `ORDER BY` value (e.g. `RANGE BETWEEN INTERVAL '7 DAYS' PRECEDING AND CURRENT ROW`).
- Default frame when `ORDER BY` is present: `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`.
- To compute a running total vs an entire partition total:
  ```sql
  -- Running total:
  SUM(amount) OVER (PARTITION BY customer_id ORDER BY order_date ROWS UNBOUNDED PRECEDING)
  -- Total across partition:
  SUM(amount) OVER (PARTITION BY customer_id)
  ```
