# SQL Aggregations, Grouping Sets, and NULL Semantics

## Standard Aggregations and Filtering
- **`GROUP BY`**: Groups rows sharing identical values in specified columns into summary rows.
- **`HAVING` vs. `WHERE`**:
  - `WHERE` filters rows *before* aggregation takes place.
  - `HAVING` filters aggregated groups *after* group computation (e.g. `HAVING COUNT(*) > 5`).

## Advanced Multi-Dimensional Grouping
1. **`GROUPING SETS`**: Generates aggregations for multiple specific column combinations in a single pass:
   ```sql
   SELECT department_id, job_title, AVG(salary)
   FROM employees
   GROUP BY GROUPING SETS ((department_id, job_title), (department_id), ());
   ```
2. **`ROLLUP`**: Generates hierarchical subtotal and grand total aggregations:
   - `ROLLUP (region, country, city)` produces groups: `(region, country, city)`, `(region, country)`, `(region)`, and `()`.
3. **`CUBE`**: Generates all $2^N$ possible permutations of cross-tabulation groupings across $N$ columns.
4. **`GROUPING()` function**: Returns `1` if the column in the current row is aggregated as part of a subtotal/grand total, or `0` if it represents actual row data.

## NULL Value Semantics in SQL Aggregations
- Standard aggregate functions (`COUNT(col)`, `SUM(col)`, `AVG(col)`, `MIN(col)`, `MAX(col)`) ignore `NULL` entries except `COUNT(*)`.
- `COUNT(*)` counts total rows in the group, including rows where all columns are `NULL`.
- If a column contains only `NULL` values:
  - `SUM(col)` returns `NULL`.
  - `AVG(col)` returns `NULL`.
  - `COUNT(col)` returns `0`.
- In `GROUP BY`, all `NULL` values are grouped together into a single group.
- `COALESCE(val1, val2, ...)` returns the first non-null expression, useful for default value substitution during aggregation.
