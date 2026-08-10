"""Golden-fragment tests for console components and display filters.

These lock the visual contract of UI principles 1-3: statuses map 1:1 to
badges, rejection is neutral (never red), and raw values render before
normalized ones.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from jinja2 import Environment

from invoice_agents.models import EvidenceValue
from invoice_agents.ui.server import (
    build_templates,
    fmt_amount,
    fmt_dt,
    fmt_duration,
    fmt_signed,
    is_nonzero,
    json_pretty,
    middle,
    tone,
)


@pytest.fixture(scope="module")
def env() -> Environment:
    return build_templates().env


def render_macro(env: Environment, call: str, **context: object) -> str:
    template = env.from_string("{% import '_macros.html' as m %}" + call)
    return template.render(**context)


def test_status_badges_map_one_to_one(env: Environment) -> None:
    for status, expected_tone in (
        ("SUCCEEDED", "tone-ok"),
        ("NEEDS_HUMAN", "tone-warn"),
        ("FAILED", "tone-fail"),
        ("INCOMPLETE", "tone-pause"),
    ):
        html = render_macro(env, "{{ m.status_badge(value) }}", value=status)
        assert status in html, "the badge must carry its literal status word"
        assert expected_tone in html


def test_rejection_renders_neutral_never_red(env: Environment) -> None:
    html = render_macro(env, "{{ m.decision_badge('REJECT') }}")
    assert "REJECT" in html
    assert "tone-pause" in html
    assert "tone-fail" not in html


def test_duplicate_payment_is_neutral(env: Environment) -> None:
    html = render_macro(env, "{{ m.payment_badge('DUPLICATE') }}")
    assert "tone-pause" in html
    assert "tone-fail" not in html


def test_missing_value_badges_render_placeholder(env: Environment) -> None:
    for call in (
        "{{ m.status_badge(none) }}",
        "{{ m.decision_badge(none) }}",
        "{{ m.payment_badge(none) }}",
    ):
        html = render_macro(env, call)
        assert "badge" not in html
        assert "-" in html


def test_provenance_row_orders_raw_before_normalized(env: Environment) -> None:
    ev = EvidenceValue(
        raw_value="$ 15 000,00",
        normalized_value="15000.00",
        normalization="stripped grouping and symbol",
        ambiguity="thousands separator was ambiguous",
    )
    html = render_macro(env, "<table>{{ m.provenance_row('Amount', ev) }}</table>", ev=ev)
    assert html.index("$ 15 000,00") < html.index("15000.00")
    assert "stripped grouping and symbol" in html
    assert "thousands separator was ambiguous" in html, "ambiguity is always visible"


def test_provenance_row_accepts_bundle_dicts(env: Environment) -> None:
    ev = {
        "raw_value": "Jan 30 2026",
        "normalized_value": "2026-01-30",
        "normalization": "dateutil parse to ISO-8601",
        "ambiguity": None,
    }
    html = render_macro(env, "<table>{{ m.provenance_row('Invoice date', ev) }}</table>", ev=ev)
    assert html.index("Jan 30 2026") < html.index("2026-01-30")


def test_delta_cell_highlights_iff_nonzero(env: Environment) -> None:
    zero = render_macro(env, "<tr>{{ m.delta_cell('0.00') }}</tr>")
    assert "delta-zero" in zero and "delta-bad" not in zero
    positive = render_macro(env, "<tr>{{ m.delta_cell('110.00') }}</tr>")
    assert "delta-bad" in positive
    assert "+110.00" in positive, "deltas carry an explicit sign and amount"
    negative = render_macro(env, "<tr>{{ m.delta_cell('-50.00') }}</tr>")
    assert "delta-bad" in negative and "-50.00" in negative
    missing = render_macro(env, "<tr>{{ m.delta_cell(none) }}</tr>")
    assert "delta-bad" not in missing and "delta-zero" not in missing


def test_preflight_strip_shows_stop_reason_and_fix(env: Environment) -> None:
    from invoice_agents.ui.preflight import PreflightItem, PreflightReport

    report = PreflightReport(
        items=[
            PreflightItem(name="inventory DB", ok=True, detail="schema v1, integrity ok"),
            PreflightItem(
                name="workflow DB",
                ok=False,
                detail="required database does not exist: workflow.db",
                stop_reason="DATABASE_MISSING",
                fix_command="uv run python -m invoice_agents.db migrate --db workflow.db --kind workflow",
            ),
        ]
    )
    template = env.from_string("{% import '_preflight.html' as pf %}{{ pf.strip(preflight) }}")
    html = template.render(preflight=report)
    assert "DATABASE_MISSING" in html
    assert "required database does not exist" in html
    assert "uv run python -m invoice_agents.db migrate --db workflow.db --kind workflow" in html
    assert "failing" in html


def test_json_disclosure_pretty_prints_payload_strings(env: Environment) -> None:
    html = render_macro(
        env,
        "{{ m.json_disclosure(payload) }}",
        payload='{"b": 1, "a": {"nested": true}}',
    )
    assert "<details" in html
    assert "&#34;a&#34;" in html or '"a"' in html
    assert html.index('"a"' if '"a"' in html else "&#34;a&#34;") >= 0


def test_display_filters() -> None:
    assert fmt_amount("15000.00") == "15,000.00"
    assert fmt_amount("-250") == "-250"
    assert fmt_amount(None) == "-"
    assert fmt_amount("not-a-number") == "not-a-number"
    assert fmt_signed("110.00") == "+110.00"
    assert fmt_signed("-50.00") == "-50.00"
    assert fmt_signed("0.00") == "0.00"
    assert is_nonzero("0.00") is False
    assert is_nonzero("110.00") is True
    assert is_nonzero(None) is False
    assert is_nonzero("unparseable") is True, "an unparseable delta stays visible"
    assert middle("a" * 40).startswith("a" * 10)
    assert "…" in middle("a" * 40)
    assert middle("short") == "short"
    assert fmt_dt("2026-08-06T15:14:13.021491+00:00") == "2026-08-06 15:14:13 UTC"
    assert fmt_duration("2026-08-06T15:14:13+00:00", "2026-08-06T15:15:54+00:00") == "1m 41s"
    assert fmt_duration("2026-08-06T15:14:13+00:00", None) == "-"
    assert json_pretty('{"z": 1}').startswith("{")
    assert json_pretty("not json") == "not json"


def test_tone_defaults_to_neutral_for_unknown_values() -> None:
    assert tone("case", "SOMETHING_NEW") == "pause"
    assert tone("unknown-kind", "X") == "pause"


def test_templates_have_no_inline_style_declarations() -> None:
    template_dir = Path(__file__).resolve().parents[2] / "src/invoice_agents/ui/templates"
    violations = {
        template.name: re.findall(r"\bstyle\s*=", template.read_text(encoding="utf-8"), re.I)
        for template in template_dir.glob("*.html")
    }
    assert {name: matches for name, matches in violations.items() if matches} == {}


def test_htmx_indicator_configuration_is_csp_safe_and_static() -> None:
    static_dir = Path(__file__).resolve().parents[2] / "src/invoice_agents/ui/static"
    base = (
        Path(__file__).resolve().parents[2] / "src/invoice_agents/ui/templates/base.html"
    ).read_text(encoding="utf-8")
    css = (static_dir / "app.css").read_text(encoding="utf-8")
    config_position = base.index('name="htmx-config"')
    script_position = base.index('src="/static/htmx.min.js"')
    assert config_position < script_position
    assert '"includeIndicatorStyles":false' in base
    assert ".htmx-indicator" in css
    assert ".htmx-request .htmx-indicator" in css
    assert "visibility: hidden" in css
    assert "visibility: visible" in css
