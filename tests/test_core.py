# test_core.py
# basic tests to confirm the library wires together correctly.
# does not test dictionary content — only that the pipeline runs
# and returns the right types with the right fields populated.

import pytest
from pathlib import Path
from dscl import DSCL, ConstrainedContext, ValidationResult


DICTIONARY_PATH = Path(__file__).parent.parent / "dictionary"


@pytest.fixture
def dscl():
    return DSCL(dictionary_path=DICTIONARY_PATH)


# ── prepare() ─────────────────────────────────────────────────────────────────

def test_prepare_returns_constrained_context(dscl):
    result = dscl.prepare("Write a summary of the quarterly financial report.")
    assert isinstance(result, ConstrainedContext)


def test_prepare_populates_all_fields(dscl):
    result = dscl.prepare("Write a summary of the quarterly financial report.")
    assert result.narrative
    assert result.domain
    assert result.fk_target
    assert result.register
    assert result.system_prompt
    assert result.prompt


def test_prepare_system_prompt_contains_domain(dscl):
    result = dscl.prepare("Write a summary of the quarterly financial report.")
    assert result.domain in result.system_prompt


def test_prepare_manual_domain_override(dscl):
    result = dscl.prepare(
        "Draft a liability clause.",
        domain="legal_contract"
    )
    assert result.domain == "legal_contract"


def test_prepare_manual_audience_override(dscl):
    result = dscl.prepare(
        "Explain the procedure to the patient.",
        domain="medical",
        audience="general_public"
    )
    assert isinstance(result, ConstrainedContext)


# ── validate() ────────────────────────────────────────────────────────────────

def test_validate_returns_validation_result(dscl):
    result = dscl.validate("The report was completed on time.")
    assert isinstance(result, ValidationResult)


def test_validate_populates_fk_grade(dscl):
    result = dscl.validate("The report was completed on time.")
    assert isinstance(result.fk_grade, float)
    assert result.fk_grade >= 0.0


def test_validate_violations_is_list(dscl):
    result = dscl.validate("The report was completed on time.")
    assert isinstance(result.violations, list)


def test_validate_fixes_applied_is_list(dscl):
    result = dscl.validate("The report was completed on time.")
    assert isinstance(result.fixes_applied, list)


def test_validate_passed_is_bool(dscl):
    result = dscl.validate("The report was completed on time.")
    assert isinstance(result.passed, bool)


def test_validate_standalone_without_prepare(dscl):
    # validate() should work without calling prepare() first
    result = dscl.validate("Short and clear output.", domain="plain_summary")
    assert isinstance(result, ValidationResult)


# ── prepare_and_validate() ────────────────────────────────────────────────────

def test_prepare_and_validate_returns_both(dscl):
    context, result = dscl.prepare_and_validate(
        narrative="Write a plain summary of the terms.",
        model_output="The terms are clear and easy to follow."
    )
    assert isinstance(context, ConstrainedContext)
    assert isinstance(result, ValidationResult)


def test_prepare_and_validate_domain_is_consistent(dscl):
    context, result = dscl.prepare_and_validate(
        narrative="Write a plain summary of the terms.",
        model_output="The terms are clear and easy to follow."
    )
    assert result.domain == context.domain


# ── fk scoring ────────────────────────────────────────────────────────────────

def test_short_simple_text_scores_low_fk(dscl):
    result = dscl.validate("The cat sat. The dog ran. It was fast.")
    assert result.fk_grade < 6.0


def test_complex_text_scores_higher_fk(dscl):
    complex_text = (
        "The implementation of organizational restructuring initiatives "
        "necessitates comprehensive evaluation of interdepartmental "
        "communication frameworks and their subsequent utilization across "
        "administrative hierarchies."
    )
    result = dscl.validate(complex_text)
    assert result.fk_grade > 10.0


# ── loader path ───────────────────────────────────────────────────────────────

def test_invalid_dictionary_path_raises():
    with pytest.raises(FileNotFoundError):
        DSCL(dictionary_path="/nonexistent/path/dictionary")