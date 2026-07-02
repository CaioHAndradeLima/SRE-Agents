# Database Connection Pool Exhaustion

## Symptoms
- Errors like `TimeoutError: could not acquire connection from pool`.
- Rising request latency across many endpoints of a service at once.
- Database `active_connections` near the configured pool limit.

## Likely causes
- A traffic surge exceeding pool capacity.
- A connection leak (connections not returned to the pool).
- A slow query holding connections open.

## Diagnosis steps
1. Check `active_connections` and `db_query_p99_ms` for the service's database.
2. Search logs for `pool`, `acquire`, and `TimeoutError`.
3. Identify slow queries; check for a recent change to query patterns.

## Remediation
- Short term: scale the service horizontally or increase the pool size.
- If a leak is suspected from a recent deploy, roll back that deploy.
- Kill long-running queries that are starving the pool.
