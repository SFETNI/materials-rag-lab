import pytest

from materials_rag.ingestion.scientific_computation import (
    ComputationRequest,
    ComputationValidationError,
    build_structured_data_catalog,
    filter_records,
    validate_computation_request,
)


def test_catalog_exposes_typed_operations() -> None:
    catalog = build_structured_data_catalog()
    descriptor = catalog["nist_experiment_records"]
    assert "SUMMARIZE" in descriptor.supported_operations
    assert descriptor.available_fields["stress_level_mpa"].unit == "MPa"


def test_filter_records_has_no_expression_language() -> None:
    records = [{"id": "a", "stress": 400}, {"id": "b", "stress": 600}]
    selected = filter_records(records, ({"field": "stress", "op": "GE", "value": 500},))
    assert selected == [{"id": "b", "stress": 600}]


def test_cycles_to_failure_requires_failure_only_policy() -> None:
    request = ComputationRequest(
        request_id="demo",
        operation="SUMMARIZE",
        dataset_id="nist_experiment_records",
        numeric_field="cycles_to_failure",
        status_filter="include_runouts",
        requested_statistics=("median",),
    )
    with pytest.raises(ComputationValidationError, match="failure_only"):
        validate_computation_request(request, build_structured_data_catalog())

