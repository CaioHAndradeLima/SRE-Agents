# Checkout 5xx / Payment Errors

## Symptoms
- Spike in `POST /checkout` returning 500 Internal Server Error.
- Elevated checkout `error_rate` and `p99_latency_ms`.
- Logs showing exceptions in `PaymentClient` (e.g. NullPointerException in
  `PaymentClient.charge()`), or payment gateway timeouts.

## Likely causes
- A recent deploy to the `checkout` service introduced a regression in payment
  handling (most common). Check the latest deploy and the commits it shipped.
- Payment gateway provider degradation (rarer; correlate with provider status).

## Diagnosis steps
1. Check `error_rate` and `p99_latency_ms` for `checkout` over the last hour.
2. Search logs for `PaymentClient`, `NullPointerException`, and `500`.
3. List recent deploys for `checkout`; note the version that shipped just before
   the spike and the commit SHA.
4. Inspect the failing CI run and `git blame` the payment files.

## Remediation
- **If a recent deploy correlates with the spike, roll back `checkout` to the
  last known-good version.** This is the standard, approved mitigation.
- Do NOT fail over to another region for a code regression — failover does not
  fix bad code and risks broader impact.
- After rollback, confirm `error_rate` returns to baseline before resolving.
