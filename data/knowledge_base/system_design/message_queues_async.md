# Asynchronous Processing, Message Queues, and Stream Processing

## Message Queue Patterns
- **Point-to-Point (Queue)**: Each message is consumed by exactly one worker in a consumer pool (e.g. RabbitMQ work queues, AWS SQS). Ideal for task offloading (image processing, email dispatch).
- **Publish-Subscribe (Pub/Sub)**: Messages published to a topic are broadcast to all subscribed consumer groups (e.g. Apache Kafka, Google Cloud Pub/Sub, Redis Pub/Sub).

## Apache Kafka Architecture
- **Distributed Commit Log**: Append-only sequential log stored on disk; reads and writes run in $O(1)$ sequential I/O.
- **Topics and Partitions**: Topics are split into ordered, immutable partitions distributed across broker clusters.
- **Offset Management**: Consumers track their read progress via offsets committed to the `__consumer_offsets` topic.
- **Consumer Groups**: Each partition within a topic is consumed by at most one consumer instance within the same group, enabling linear horizontal read scaling.

## Delivery Guarantees
1. **At-Most-Once**: Messages may be lost but are never re-delivered (consumer commits offset *before* processing message).
2. **At-Least-Once**: Messages are guaranteed not to be lost, but duplicates may occur (consumer commits offset *after* processing message). Requires **idempotent** downstream operations.
3. **Exactly-Once Semantics (EOS)**: Uses two-phase commit transactions or idempotent producer IDs ($PID + \text{Sequence Number}$) to ensure state updates occur exactly once end-to-end.

## Dead Letter Queues (DLQ)
When a poison message repeatedly fails processing after $N$ retry attempts with exponential backoff, it is routed to a Dead Letter Queue to prevent pipeline stalling while alerting engineers for debugging.
