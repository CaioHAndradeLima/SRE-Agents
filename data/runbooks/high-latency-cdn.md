# High Latency from CDN / Edge

## Symptoms
- Increased `p99_latency_ms` for static assets or cached endpoints.
- Elevated cache miss rate at the edge.
- Users in specific regions report slowness while others are fine.

## Likely causes
- A CDN provider incident in one or more edge regions.
- A cache configuration change that lowered the hit rate.
- Origin overload causing slow fills.

## Diagnosis steps
1. Compare latency by region; isolate whether it is region-specific.
2. Check cache hit rate before/after any recent config change.
3. Review the CDN provider status page.

## Remediation
- If a single edge region is degraded, fail over traffic to a healthy region.
- Revert a recent cache-configuration change if it dropped the hit rate.
- Scale origin capacity if origin fills are the bottleneck.
