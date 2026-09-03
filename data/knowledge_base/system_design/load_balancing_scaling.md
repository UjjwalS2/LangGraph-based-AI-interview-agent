# Load Balancing, Horizontal Scaling, and Consistent Hashing

## Load Balancing Layers
- **Layer 4 Load Balancing (Transport Layer - TCP/UDP)**: Routes traffic based on IP address and port without inspecting application packet content (e.g. AWS NLB, IPVS). Ultra-high throughput and low CPU overhead.
- **Layer 7 Load Balancing (Application Layer - HTTP/HTTPS/gRPC)**: Inspects HTTP headers, cookies, URL paths, and payload content (e.g. Nginx, Envoy, AWS ALB). Enables smart routing, SSL termination, and content-based microservice dispatch.

## Load Balancing Algorithms
1. **Round Robin**: Distributes requests sequentially across all nodes.
2. **Weighted Round Robin**: Routes traffic proportionally according to server capacity/specifications.
3. **Least Connections**: Dispatches new requests to the instance with the fewest active concurrent connections.
4. **IP Hash**: Hashes client IP to ensure sticky session affinity.

## Consistent Hashing
Traditional modulo hashing `server_index = hash(key) % N` breaks catastrophically when servers are added or removed, remapping almost $100\%$ of keys across the cluster.
- **Ring Architecture**: Maps both server nodes and cache keys onto a circular hash ring $[0, 2^{32}-1]$.
- A key is routed clockwise to the first server node encountered on the ring.
- **Adding/Removing a Node**: Only keys between the affected node and its predecessor need to be migrated (amortized $\frac{K}{N}$ keys relocated).
- **Virtual Nodes (Vnodes)**: Maps each physical server to multiple positions (e.g. 100-500 vnodes) on the ring to prevent hotspot imbalances and guarantee uniform load distribution.

## Horizontal vs. Vertical Scalability
- **Vertical Scaling (Scale-Up)**: Increasing CPU, RAM, or NVMe storage on a single machine. Limited by hardware boundaries, high cost, and single point of failure (SPOF).
- **Horizontal Scaling (Scale-Out)**: Adding more commodity nodes to the server pool. Requires stateless application tiers, external session stores (Redis), and distributed data layers.
