"""Package-resource boundaries for runtime database migrations."""

from invoice_agents.db.core import DatabaseKind, _migration_resources


def test_each_database_kind_has_packaged_migrations() -> None:
    """Removing a shipped migration must make database setup impossible."""

    assert [item.name for item in _migration_resources(DatabaseKind.INVENTORY)] == [
        "001_initial.sql"
    ]
    assert [item.name for item in _migration_resources(DatabaseKind.WORKFLOW)] == [
        "001_initial.sql",
        "002_review_sequence.sql",
        "003_execution_fencing.sql",
        "004_execution_token_grammar.sql",
        "005_result_artifact_bindings.sql",
        "006_durable_admission.sql",
        "007_critique_cycles.sql",
    ]
