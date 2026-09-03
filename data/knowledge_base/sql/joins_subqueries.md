# SQL Joins, Subqueries, and Common Table Expressions (CTEs)

## Relational Joins
Joins combine rows from two or more tables based on related columns:
- **INNER JOIN**: Returns records that have matching values in both tables.
- **LEFT (OUTER) JOIN**: Returns all records from the left table and matched records from the right table; unmatched right-side attributes evaluate to `NULL`.
- **RIGHT (OUTER) JOIN**: Returns all records from the right table and matched records from the left table.
- **FULL (OUTER) JOIN**: Returns all records when there is a match in either left or right table.
- **CROSS JOIN**: Produces the Cartesian product ($N \times M$ rows) of both tables.
- **SELF JOIN**: A regular join where a table is joined with itself (e.g. employee-manager hierarchies).

## Subqueries vs. CTEs
- **Non-Correlated Subquery**: An independent query evaluated once before the outer query executes.
- **Correlated Subquery**: A subquery that references columns from the outer query table, evaluated repeatedly once for each candidate row in the outer query, often incurring $O(N \times M)$ runtime if not unnested by the query optimizer.
- **Common Table Expressions (CTEs)**:
  - Defined using the `WITH` clause:
    ```sql
    WITH RankedEmployees AS (
      SELECT id, department_id, salary,
             DENSE_RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) as rank
      FROM employees
    )
    SELECT * FROM RankedEmployees WHERE rank <= 3;
    ```
  - Enhances readability, reusability within the same statement, and modular query debugging.
  - Modern query planners (PostgreSQL 12+, SQL Server) inline non-recursive CTEs unless declared `AS MATERIALIZED`.

## Recursive CTEs
Recursive CTEs compute hierarchical or graph traversals:
- Consist of an Anchor member, `UNION ALL`, and a Recursive member referencing the CTE name.
- Recursion terminates when the recursive member yields an empty result set.
