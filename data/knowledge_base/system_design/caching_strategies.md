# System Design: Caching Patterns and Invalidation Strategies

## Caching Access Patterns
1. **Cache-Aside (Lazy Loading)**:
   - Application first reads from cache. On miss, application reads from database, stores result in cache, and returns it.
   - Pros: Only requested data is cached; resilient to cache node failures.
   - Cons: Cache miss penalty on first access; potential data staleness if writes bypass cache.
2. **Read-Through**:
   - Application interacts only with cache. On miss, the cache transparently loads data from the database.
3. **Write-Through**:
   - Application writes to cache, and the cache synchronously writes to the database before acknowledging success.
   - Pros: High data consistency; reads are always warm.
   - Cons: High write latency (double write penalty).
4. **Write-Behind (Write-Back)**:
   - Application writes to cache immediately; cache asynchronously flushes batches to the database.
   - Pros: Ultra-fast write throughput.
   - Cons: Risk of permanent data loss if cache crashes before flushing dirty pages.

## Eviction Policies
- **LRU (Least Recently Used)**: Discards items not accessed for the longest duration (implemented via Doubly Linked List + Hash Map in $O(1)$).
- **LFU (Least Frequently Used)**: Discards items with lowest access counts.
- **TTL (Time to Live)**: Expiration timestamps ensuring periodic background refresh.

## Cache Failure Modes and Mitigations
- **Cache Avalanche**: Large number of keys expire simultaneously, causing all subsequent queries to hit the database at once. *Mitigation*: Add random jitter/delta to expiration TTLs (`TTL + rand(0, 300)`).
- **Cache Stampede / Dogpiling**: Multiple concurrent requests for an expired popular key trigger redundant expensive database computations simultaneously. *Mitigation*: Mutex locking / probabilistic early expiration (XFetch algorithm).
- **Cache Penetration**: Malicious or invalid queries for keys that do not exist in DB or cache bypass cache entirely. *Mitigation*: Cache null values with short TTL, or use **Bloom Filters** at cache entrance.
