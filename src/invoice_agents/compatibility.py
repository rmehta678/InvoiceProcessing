"""Opt-in live contracts proving AutoGen 0.7.5 behavior against exact xAI Grok 4.5."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import openai
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.messages import (
    HandoffMessage,
    StructuredMessage,
    ToolCallExecutionEvent,
)
from autogen_agentchat.teams import Swarm
from autogen_core.models import UserMessage
from autogen_core.tools import FunctionTool
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel, ConfigDict

from invoice_agents.agents.team import create_model_client
from invoice_agents.config import XAI_BASE_URL, XAI_MODEL, Settings
from invoice_agents.db.core import DatabaseKind, migrate_database
from invoice_agents.errors import ErrorCategory
from invoice_agents.models import UsageSummary
from invoice_agents.observability.audit import AuditRecorder, safe_provider_request_id
from invoice_agents.orchestration import _error_record


class ProbeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: int
    explanation: str


class ContractCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    evidence: str


def _stable_evidence(category: str, status: str | int, request_id: object = None) -> str:
    """Encode only bounded contract classifications, never provider-controlled prose."""

    validated_request_id = safe_provider_request_id(request_id)
    return (
        f"category={category}; status={status}; provider_request_id="
        f"{validated_request_id or '<absent>'}"
    )


async def run_live_contracts(settings: Settings) -> list[ContractCheck]:
    """Run paid live checks; callers must opt in and skips are never reported as passes."""

    checks: list[ContractCheck] = []
    client = create_model_client(settings)
    try:
        basic = await client.create(
            [UserMessage(content="Reply with exactly the word OK.", source="compatibility_probe")]
        )
        checks.append(
            ContractCheck(
                name="authentication_basic_completion",
                passed=bool(basic.content) and basic.finish_reason is not None,
                evidence=_stable_evidence(
                    "MODEL_COMPLETION",
                    "COMPLETE"
                    if bool(basic.content) and basic.finish_reason is not None
                    else "INCOMPLETE",
                ),
            )
        )

        calls: list[tuple[int, int]] = []

        async def add_exact(left: int, right: int) -> int:
            """Add two integers and record the typed call."""

            calls.append((left, right))
            return left + right

        typed_agent = AssistantAgent(
            "typed_tool_probe",
            client,
            tools=[
                FunctionTool(
                    add_exact,
                    description="Add two required integers exactly.",
                    strict=True,
                )
            ],
            system_message=(
                "Call add_exact with 19 and 23 exactly once. Return the exact tool result and do not "
                "calculate it yourself."
            ),
            reflect_on_tool_use=False,
            max_tool_iterations=1,
        )
        typed_result = await typed_agent.run(task="Execute the strict typed function-call probe.")
        tool_result_seen = any(
            isinstance(message, ToolCallExecutionEvent)
            and any(not item.is_error and item.name == "add_exact" for item in message.content)
            for message in typed_result.messages
        )
        structured_agent = AssistantAgent(
            "structured_probe",
            client,
            system_message=(
                "Return ProbeOutput using the already-recorded tool evidence. Do not add fields."
            ),
            output_content_type=ProbeOutput,
        )
        structured_result = await structured_agent.run(
            task=f"The strict add_exact tool result is {calls[-1][0] + calls[-1][1]}. Reflect on it."
        )
        structured_messages = [
            message
            for message in structured_result.messages
            if isinstance(message, StructuredMessage)
        ]
        structured_ok = bool(structured_messages) and structured_messages[-1].content.result == 42
        checks.append(
            ContractCheck(
                name="typed_tool_and_structured_output",
                passed=tool_result_seen and structured_ok and calls == [(19, 23)],
                evidence=(
                    f"tool_calls={calls}; tool_result_event={tool_result_seen}; "
                    f"structured_after_tool={structured_ok}"
                ),
            )
        )

        sequence: list[int] = []

        async def record_step(step: int) -> dict[str, Any]:
            """Record a sequential stateful tool step."""

            sequence.append(step)
            return {"recorded": step, "sequence": list(sequence)}

        sequential_agent = AssistantAgent(
            "sequential_probe",
            client,
            tools=[record_step],
            system_message=(
                "Call record_step(step=1), inspect its result, then call record_step(step=2). Calls must be "
                "sequential. After both, say SEQUENCE_DONE."
            ),
            reflect_on_tool_use=False,
            max_tool_iterations=3,
        )
        await sequential_agent.run(task="Run both stateful tool steps in order.")
        checks.append(
            ContractCheck(
                name="sequential_tool_iterations_parallel_disabled",
                passed=sequence == [1, 2],
                evidence=f"observed_sequence={sequence}",
            )
        )

        async def first_probe() -> str:
            """Return first-agent evidence."""

            return "first-agent-tool-ok"

        async def second_probe() -> str:
            """Return second-agent evidence."""

            return "second-agent-tool-ok"

        first = AssistantAgent(
            "handoff_first",
            client,
            tools=[first_probe],
            handoffs=["handoff_second"],
            system_message="Call first_probe, then hand off to handoff_second.",
            reflect_on_tool_use=True,
            max_tool_iterations=2,
        )
        second = AssistantAgent(
            "handoff_second",
            client,
            tools=[second_probe],
            system_message="Call second_probe, then say HANDOFF_DONE.",
            reflect_on_tool_use=True,
            max_tool_iterations=2,
        )
        handoff_team = Swarm(
            [first, second], termination_condition=TextMentionTermination("HANDOFF_DONE")
        )
        handoff_result = await handoff_team.run(task="Prove a model-driven Swarm handoff.")
        handoff_seen = any(
            isinstance(message, HandoffMessage) for message in handoff_result.messages
        )
        checks.append(
            ContractCheck(
                name="swarm_handoff_and_agent_names",
                passed=handoff_seen,
                evidence=(
                    f"handoff_seen={handoff_seen}; include_name_in_message=false; "
                    "add_name_prefixes=true"
                ),
            )
        )

        review_calls: list[str] = []

        async def review_probe(stage: str) -> str:
            """Record pre/post review tool use."""

            review_calls.append(stage)
            return stage

        reviewer_agent = AssistantAgent(
            "review_probe_agent",
            client,
            tools=[review_probe],
            handoffs=["human_reviewer"],
            system_message=(
                "If there is no human approval message, call review_probe(stage='before'), then hand off to "
                "human_reviewer. If the latest message is from human_reviewer and says approved, you MUST "
                "first call the review_probe tool with stage='after' and only after that tool result may "
                "you say REVIEW_RESUMED_DONE. Saying REVIEW_RESUMED_DONE without the stage='after' tool "
                "call is a contract violation."
            ),
            reflect_on_tool_use=True,
            max_tool_iterations=2,
        )
        review_team = Swarm(
            [reviewer_agent],
            termination_condition=(
                HandoffTermination("human_reviewer") | TextMentionTermination("REVIEW_RESUMED_DONE")
            ),
        )
        first_stop = await review_team.run(task="Start the persisted review probe.")
        saved = await review_team.save_state()
        await review_team.load_state(saved)
        resumed = await review_team.run(
            task=HandoffMessage(
                source="human_reviewer",
                target="review_probe_agent",
                # The termination phrase must never appear here: TextMentionTermination
                # scans injected task messages too and would stop the team unrun.
                content=(
                    "approved; resume the probe: call review_probe(stage='after') now, then emit "
                    "the completion phrase from your instructions"
                ),
            )
        )
        resume_done = any(
            "REVIEW_RESUMED_DONE" in str(getattr(message, "content", ""))
            for message in resumed.messages
        )
        checks.append(
            ContractCheck(
                name="handoff_stop_save_load_human_resume",
                passed=(
                    "human_reviewer" in str(first_stop.stop_reason)
                    and bool(saved)
                    and resume_done
                    and review_calls == ["before", "after"]
                ),
                evidence=(
                    f"handoff_observed={'human_reviewer' in str(first_stop.stop_reason)}; "
                    f"state_saved={bool(saved)}; review_call_count={len(review_calls)}; "
                    f"resumed={resume_done}"
                ),
            )
        )

        async def exploding_tool() -> str:
            """Raise a known exception to verify the tool error remains observable."""

            raise RuntimeError("compatibility-tool-sentinel")

        exception_agent = AssistantAgent(
            "exception_probe",
            client,
            tools=[exploding_tool],
            system_message="Call exploding_tool once, then report its exact failure without success wording.",
            reflect_on_tool_use=False,
            max_tool_iterations=1,
        )
        exception_result = await exception_agent.run(task="Run the exception propagation probe.")
        error_events = [
            message
            for message in exception_result.messages
            if isinstance(message, ToolCallExecutionEvent)
            and any(item.is_error for item in message.content)
        ]
        error_text = " ".join(
            item.content for event in error_events for item in event.content if item.is_error
        )
        checks.append(
            ContractCheck(
                name="tool_exception_visibility",
                passed="compatibility-tool-sentinel" in error_text,
                evidence=f"error_event_count={len(error_events)}; sentinel_retained={'compatibility-tool-sentinel' in error_text}",
            )
        )

        checks.append(await _server_echoed_model_identity(settings))
        checks.append(await _live_invalid_key_rejection())
        checks.append(await _telemetry_span_and_usage_capture(client))
        checks.append(await _structured_output_rejection_live(settings))
    finally:
        await client.close()
    return checks


async def _server_echoed_model_identity(settings: Settings) -> ContractCheck:
    """§11 item 2, closed properly: the server, not our constants, names the model.

    Test-only direct client; the runtime path stays the single AutoGen client.
    """

    direct = openai.AsyncOpenAI(base_url=XAI_BASE_URL, api_key=settings.provider_key())
    try:
        raw = await direct.chat.completions.with_raw_response.create(
            model=XAI_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly the word OK."}],
        )
        completion = raw.parse()
        request_id = safe_provider_request_id(raw.headers.get("x-request-id"))
        echoed = completion.model or ""
        matched = echoed.startswith("grok-4.5")
        return ContractCheck(
            name="server_echoed_model_identity",
            passed=matched,
            evidence=_stable_evidence(
                "MODEL_IDENTITY",
                "MATCHED" if matched else "MISMATCHED",
                request_id,
            ),
        )
    finally:
        await direct.close()


async def _live_invalid_key_rejection() -> ContractCheck:
    """A deliberately incorrect key must be explicitly rejected, never accepted.

    Measured live 2026-08-06: xAI signals incorrect credentials as HTTP 400
    code=invalid-argument ("Incorrect API key provided.") for both sk- and
    xai-shaped keys - it does not emit the OpenAI-conventional 401, so the SDK
    raises BadRequestError, which _error_record keeps as a hard PROVIDER failure.
    This check asserts that observed contract; ADR-002 records the deviation. The
    AuthenticationError -> AUTHENTICATION mapping stays covered synthetically for
    providers/endpoints that do send 401.
    """

    invalid = openai.AsyncOpenAI(
        base_url=XAI_BASE_URL,
        api_key="xai-invalid-contract-probe-00000000000000000000",
        max_retries=0,
    )
    try:
        await invalid.chat.completions.create(
            model=XAI_MODEL,
            messages=[{"role": "user", "content": "This request must be rejected."}],
        )
        return ContractCheck(
            name="live_invalid_key_rejection",
            passed=False,
            evidence=_stable_evidence("AUTHENTICATION", "UNEXPECTED_ACCEPT"),
        )
    except openai.APIStatusError as exc:
        record = _error_record(exc)
        rejected_as_bad_credential = (
            exc.status_code in (400, 401)
            and record.category in (ErrorCategory.PROVIDER, ErrorCategory.AUTHENTICATION)
            and record.stop_reason in ("PROVIDER_REQUEST_FAILED", "PROVIDER_AUTHENTICATION_FAILED")
        )
        return ContractCheck(
            name="live_invalid_key_rejection",
            passed=rejected_as_bad_credential,
            evidence=_stable_evidence(
                str(record.category),
                exc.status_code,
                record.provider_request_id or safe_provider_request_id(exc.request_id),
            ),
        )
    except openai.OpenAIError as exc:
        return ContractCheck(
            name="live_invalid_key_rejection",
            passed=False,
            evidence=_stable_evidence(
                "OPENAI_ERROR",
                "NO_HTTP_STATUS",
                safe_provider_request_id(getattr(exc, "request_id", None)),
            ),
        )
    finally:
        await invalid.close()


async def _telemetry_span_and_usage_capture(client: Any) -> ContractCheck:
    """One live probe under an in-memory span exporter proves telemetry really flows.

    Audit spans must carry the case/agent attributes and the streamed usage must
    land in UsageSummary through the same accumulation the case runtime uses.
    """

    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        trace.set_tracer_provider(TracerProvider())
        provider = trace.get_tracer_provider()
    exporter = InMemorySpanExporter()
    cast(TracerProvider, provider).add_span_processor(SimpleSpanProcessor(exporter))
    case_id = "contract_telemetry_probe"
    temp_dir = Path(tempfile.mkdtemp(prefix="contract-otel-"))
    try:
        workflow_db = temp_dir / "workflow.db"
        migrate_database(workflow_db, DatabaseKind.WORKFLOW)
        audit = AuditRecorder(workflow_db, case_id)
        probe = AssistantAgent(
            "telemetry_probe_agent",
            client,
            system_message="Reply with exactly the word OK.",
        )
        usage = UsageSummary()
        # Accumulation mirrors orchestration._stream_team: streamed model usage is the
        # only source for UsageSummary token counts.
        async for event in probe.run_stream(task="Run the telemetry capture probe."):
            model_usage = getattr(event, "models_usage", None)
            if model_usage is not None:
                usage.prompt_tokens += int(model_usage.prompt_tokens)
                usage.completion_tokens += int(model_usage.completion_tokens)
                usage.model_calls += 1
            audit.record(
                f"autogen.{type(event).__name__}",
                {"type": type(event).__name__},
                agent_name=getattr(event, "source", None),
            )
        spans = exporter.get_finished_spans()
        attributed = [
            span
            for span in spans
            if span.name.startswith("autogen.")
            and (span.attributes or {}).get("invoice.case_id") == case_id
            and (span.attributes or {}).get("invoice.agent") == "telemetry_probe_agent"
        ]
        return ContractCheck(
            name="telemetry_span_and_usage_capture",
            passed=bool(attributed) and usage.prompt_tokens > 0 and usage.model_calls > 0,
            evidence=(
                f"spans_total={len(spans)}; spans_with_case_and_agent={len(attributed)}; "
                f"prompt_tokens={usage.prompt_tokens}; completion_tokens="
                f"{usage.completion_tokens}; model_calls={usage.model_calls}"
            ),
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def _structured_output_rejection_live(settings: Settings) -> ContractCheck:
    """An unsupported response schema must fail explicitly, never as silent prose."""

    direct = openai.AsyncOpenAI(base_url=XAI_BASE_URL, api_key=settings.provider_key())
    try:
        await direct.chat.completions.create(
            model=XAI_MODEL,
            messages=[{"role": "user", "content": "Return the probe object."}],
            # extra_body bypasses SDK-side typing so the provider itself must rule on
            # this deliberately invalid JSON-Schema payload.
            extra_body={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "unsupported_schema_probe",
                        "strict": True,
                        "schema": {"type": "no_such_type", "properties": 17},
                    },
                }
            },
        )
        return ContractCheck(
            name="structured_output_rejection_live",
            passed=False,
            evidence=_stable_evidence("PROVIDER", "UNEXPECTED_ACCEPT"),
        )
    except openai.APIError as exc:
        status = getattr(exc, "status_code", None)
        record = _error_record(exc)
        return ContractCheck(
            name="structured_output_rejection_live",
            passed=True,
            evidence=_stable_evidence(
                str(record.category),
                status if type(status) is int else "ERROR",
                record.provider_request_id
                or safe_provider_request_id(getattr(exc, "request_id", None)),
            ),
        )
    finally:
        await direct.close()
