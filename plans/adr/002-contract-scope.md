# ADR-002: Live compatibility-contract scope and accepted synthetic residuals

Status: **Accepted**, 2026-08-06. Recorded as part of remediation phase R3
([`REMEDIATION_PLAN.md`](../REMEDIATION_PLAN.md) §5, gap G7).

Related: [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md) §11 (the fourteen compatibility
items), `src/invoice_agents/compatibility.py` (`run_live_contracts`),
[provider compatibility contract](../../Docs/REFERENCE.md#provider-compatibility-contract),
[ADR-001](001-composite-agent-tools.md).

## Context

§11 requires a contract suite proving fourteen provider-compatibility items before trusting
AutoGen 0.7.5 with xAI `grok-4.5`. The original spike proved the orchestration-level items live
but degraded four checks to local assertions: model identity was asserted against our own
constants rather than the server's echo, and the 401, telemetry, and unsupported-schema checks ran
without the network. R3 closes this by adding four live checks to `run_live_contracts`:

1. **Server-echoed model identity** - one direct `openai.AsyncOpenAI` chat call (test-only, not a
   runtime path) asserting `response.model` starts with `grok-4.5` and capturing `x-request-id`
   from the raw response headers as evidence.
2. **Live invalid-key rejection** - a deliberately incorrect key must be explicitly rejected,
   never accepted. The remediation plan assumed the OpenAI convention (`401` →
   `openai.AuthenticationError`); measured live on 2026-08-06, **xAI instead returns HTTP 400
   `code=invalid-argument` ("Incorrect API key provided.") for incorrect keys of both `sk-` and
   `xai-` shape**. The check (`live_invalid_key_rejection`) therefore asserts the observed
   contract: an explicit `BadRequestError`/`AuthenticationError` whose body identifies the
   incorrect key, mapped by `_error_record` to a hard `PROVIDER`/`AUTHENTICATION` failure - and it
   fails if the provider ever accepts the key or answers with any other shape. The
   `AuthenticationError → AUTHENTICATION/PROVIDER_AUTHENTICATION_FAILED` mapping remains covered
   synthetically for providers and endpoints that do send 401.
3. **Telemetry capture** - a probe under an in-memory OpenTelemetry span exporter asserting the
   model-call audit spans carry case/agent attributes and that streamed `usage.prompt_tokens > 0`
   reaches `UsageSummary`.
4. **Live structured-output rejection** - an intentionally unsupported response schema must
   produce an explicit provider error, never silent prose acceptance. Measured live: xAI rejects
   it with HTTP 400 "Schema validation failed".

## Decision

With R3 in place, the §11 items are proven as follows:

| §11 item | Proven | How |
|---|---|---|
| 1. Authentication + basic completion | Live | `authentication_basic_completion` check |
| 2. Exact model ID and base URL | Live | server-echoed `response.model` starts with `grok-4.5`; `x-request-id` captured |
| 3. Typed function call round trip | Live | strict typed tool call and result |
| 4. Sequential tool iterations, parallel disabled | Live | multi-step tool sequence check |
| 5. Pydantic structured output | Live | structured-output probe |
| 6. Structured output after tool use | Live | same probe, after tool evidence |
| 7. Swarm handoff | Live | two-agent handoff check |
| 8. Stop, state save/load, human resume | Live | `handoff_stop_save_load_human_resume` check |
| 9. Tool exception propagation | Live | sentinel exception surfaces in the error text |
| 10. Invalid/unsupported JSON Schema | Live | unsupported response schema rejected with an explicit provider error |
| 11. Key/401/429/timeout/retry visibility | Invalid-key rejection live (as 400 `invalid-argument`, xAI's observed signal); 401/429/timeout classes local | see residuals below |
| 12. Event logging, usage, request ID, OTel | Live | spans carry case/agent attributes; streamed `prompt_tokens > 0` lands in `UsageSummary` |
| 13. Agent-name configuration | Live | fixed `include_name_in_message=False` + `add_name_prefixes=True`, exercised by the handoff checks |
| 14. PDF-derived image input | Not applicable | the workflow does not send page images to the provider (xAI vision unused) |

### Accepted residuals (synthetic, local)

- **429 rate-limit exhaustion** and **provider timeout classes** remain synthetically tested:
  constructed SDK exceptions are driven through `_error_record`
  (`src/invoice_agents/orchestration.py`) and asserted to map to the documented categories and
  stop reasons with preserved messages and request IDs. Deliberately inducing a real 429
  exhaustion or timeout against the live endpoint is unreliable (it depends on account limits and
  provider load) and costly, so these stay local by decision rather than omission.
- **True HTTP 401 handling** also remains synthetic: xAI does not emit 401 for incorrect keys (it
  signals 400 `invalid-argument`, proven live above), so the `AuthenticationError` branch cannot
  be exercised against this endpoint without a credential state we cannot create on demand
  (e.g. a revoked-but-well-formed token).
- **Missing-key visibility** is a local preflight guarantee by design: `Settings.provider_key`
  raises `PROVIDER_PREFLIGHT_FAILED` before any network call is attempted.

## Consequences

- A skipped check is never reported as a pass: `invoice-agents contract` without `--live` prints
  `NOT RUN` and exits nonzero (exit code 2), and any failing live check makes
  `invoice-agents contract --live` exit nonzero (exit code 1).
- The "degraded to local" state of §11 items 11 (partially) is now a recorded, bounded decision
  with its rationale, instead of an undocumented gap.
- If xAI later offers a reliable, cheap way to trigger rate-limit or timeout responses on demand,
  the residuals should be promoted to live checks and this ADR superseded.
