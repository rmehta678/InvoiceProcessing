"""Explicit PDF limits shared by tests without importing an ambiguous conftest module."""

from invoice_agents.config import PdfPolicy

TEST_PDF_POLICY = PdfPolicy(
    pdf_max_pages=100,
    pdf_parse_timeout_seconds=15.0,
    pdf_worker_cpu_seconds=10,
    pdf_worker_memory_bytes=536_870_912,
    pdf_worker_result_max_bytes=4_194_304,
)
