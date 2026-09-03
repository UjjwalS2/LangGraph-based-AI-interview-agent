# API Design, Distributed Databases, and the CAP Theorem

## API Paradigms: REST vs. gRPC vs. GraphQL
- **REST (Representational State Transfer)**: Stateless HTTP/JSON communication. Standardized HTTP verbs (`GET`, `POST`, `PUT`, `DELETE`). Easy to cache, human-readable, but prone to over-fetching or under-fetching.
- **gRPC**: Binary protocol built over HTTP/2 using Protocol Buffers (`Protobuf`). Features multiplexed streaming, bidirectional RPCs, strongly-typed contracts, and $5-10\times$ lower serialization latency than JSON. Standard for internal microservice communication.
- **GraphQL**: Query language allowing clients to declare the exact schema fields required, resolving over-fetching in complex mobile/web frontends.

## Distributed Databases: Partitioning and Replication
- **Sharding (Horizontal Partitioning)**: Distributes rows across distinct physical database nodes:
  - *Range-based Sharding*: Partitions by key ranges (e.g. A-E, F-J). Risk: Hotspots on sequential keys (e.g. timestamps).
  - *Hash-based Sharding*: Partitions via `hash(shard_key) % N`. Uniform distribution, but expensive range scans.
- **Replication Topologies**:
  - *Single-Leader (Master-Replica)*: All writes go to leader; reads scale across read replicas. Asynchronous replication introduces replication lag (eventual consistency).
  - *Multi-Leader*: Writes accepted across multiple datacenters; requires conflict resolution strategies (Last-Write-Wins, CRDTs).
  - *Leaderless (Dynamo-style)*: Client writes/reads from quorums ($W + R > N$).

## CAP Theorem and PACELC
- **CAP Theorem**: In any asynchronous network subject to partitions ($P$):
  - **Consistency (CP)**: Every read receives the most recent write or an error (e.g. Spanner, HBase, ZooKeeper).
  - **Availability (AP)**: Every non-failing node returns a non-error response, but data may be stale (e.g. Cassandra, DynamoDB, CouchDB).
- **PACELC Theorem**: Extends CAP:
  - If there is a Partition ($P$), tradeoff between Availability ($A$) and Consistency ($C$).
  - Else ($E$), tradeoff between Latency ($L$) and Consistency ($C$).
