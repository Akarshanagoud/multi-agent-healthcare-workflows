"""Per-agent behaviour: routing, extraction, retrieval, validation, summary."""
import pytest

from clinicflow import (
    Document,
    DocumentType,
    EchoProvider,
    ExtractorAgent,
    KnowledgeBase,
    Passage,
    RetrieverAgent,
    RouterAgent,
    Severity,
    SummarizerAgent,
    ValidatorAgent,
    WorkflowState,
)
from clinicflow.agents.router import score_document
from clinicflow.agents.validator import parse_date

DISCHARGE = """DISCHARGE SUMMARY

Patient: Maria Alvarez
MRN: HX4471209
Admission Date: 03/02/2026
Discharge Date: 03/06/2026

Primary Diagnosis: Community acquired pneumonia

Hospital Course: Improved on antibiotics.

BP 122/78, HR 78, Temp 98.4 F
Hemoglobin 13.1 g/dL, Creatinine 0.9 mg/dL, Potassium 4.1 mEq/L

Discharge Medications: Amoxicillin 500 mg PO tid, Lisinopril 10 mg PO daily

Allergies: NKDA
"""


def state_for(text: str, doc_type: DocumentType = DocumentType.UNKNOWN) -> WorkflowState:
    return WorkflowState(document=Document(text=text, doc_type=doc_type))


# --- router ------------------------------------------------------------------
def test_router_identifies_a_discharge_summary():
    state = state_for(DISCHARGE)
    result = RouterAgent().execute(state)
    assert state.document.doc_type is DocumentType.DISCHARGE_SUMMARY
    assert result.confidence > 0.3
    assert "extractor" in result.output["route"]


def test_router_identifies_a_lab_report():
    text = "LABORATORY REPORT\nHemoglobin 13.2 g/dL  Reference Range: 12-16\nCreatinine 1.1 mg/dL"
    state = state_for(text)
    RouterAgent().execute(state)
    assert state.document.doc_type is DocumentType.LAB_REPORT


def test_router_falls_back_to_unknown_with_no_signals():
    state = state_for("Called the patient back. Left a voicemail.")
    result = RouterAgent().execute(state)
    assert state.document.doc_type is DocumentType.UNKNOWN
    assert result.confidence == 0.0
    assert any(i.code == "CF-ROUTE-001" for i in result.issues)


def test_router_flags_an_ambiguous_document():
    # Deliberately balanced signals from two types.
    text = "Reason for referral: follow-up. Reference range noted. Hemoglobin 12 g/dL. mg/dL"
    result = RouterAgent().execute(state_for(text))
    codes = {i.code for i in result.issues}
    assert codes & {"CF-ROUTE-002", "CF-ROUTE-003"}


def test_scoring_has_diminishing_returns():
    once = score_document("Reference range")
    twice = score_document("Reference range Reference range")
    key = DocumentType.LAB_REPORT
    # Repetition raises the score but by less than a second full weight.
    assert twice[key] > once[key]
    assert twice[key] < 2 * once[key]


# --- extractor ---------------------------------------------------------------
def test_extractor_pulls_identity_and_vitals():
    state = state_for(DISCHARGE, DocumentType.DISCHARGE_SUMMARY)
    result = ExtractorAgent().execute(state)
    state.add_result(result)
    assert state.fields["patient_name"].value == "Maria Alvarez"
    assert state.fields["blood_pressure"].value == "122/78"
    assert state.fields["heart_rate"].value == 78
    assert state.fields["primary_diagnosis"].value.startswith("Community acquired")


def test_extracted_names_do_not_span_lines():
    state = state_for("Patient: Grace Okafor\nMRN: HX8812340\n")
    result = ExtractorAgent().execute(state)
    state.add_result(result)
    assert state.fields["patient_name"].value == "Grace Okafor"


def test_every_field_carries_a_usable_span():
    state = state_for(DISCHARGE, DocumentType.DISCHARGE_SUMMARY)
    result = ExtractorAgent().execute(state)
    for extracted in result.fields:
        start, end = extracted.span
        assert DISCHARGE[start:end] == extracted.evidence


def test_medications_are_structured():
    state = state_for(DISCHARGE)
    result = ExtractorAgent().execute(state)
    state.add_result(result)
    medications = state.fields["medications"].value
    drugs = {entry["drug"].lower() for entry in medications}
    assert {"amoxicillin", "lisinopril"} <= drugs
    lisinopril = next(e for e in medications if e["drug"].lower() == "lisinopril")
    assert lisinopril["dose"] == 10.0
    assert lisinopril["unit"] == "mg"
    assert lisinopril["frequency"] == "daily"


def test_nkda_is_an_empty_allergy_list_not_an_allergy():
    state = state_for("Allergies: NKDA")
    result = ExtractorAgent().execute(state)
    state.add_result(result)
    assert state.fields["allergies"].value == []


def test_allergies_are_split():
    state = state_for("Allergies: Penicillin, sulfa")
    result = ExtractorAgent().execute(state)
    state.add_result(result)
    assert {a.lower() for a in state.fields["allergies"].value} == {"penicillin", "sulfa"}


def test_three_digit_lab_values_are_not_truncated():
    # A single-digit capture silently turned "41" into "4", hiding an
    # impossible value from the plausibility check.
    state = state_for("Potassium 41 mEq/L")
    result = ExtractorAgent().execute(state)
    state.add_result(result)
    assert state.fields["potassium"].value == 41.0


def test_empty_document_reports_an_error():
    result = ExtractorAgent().execute(state_for("...."))
    assert any(i.code == "CF-EXT-001" for i in result.issues)


def test_model_assist_is_skipped_without_a_provider():
    state = state_for("Some note with no labelled diagnosis field.")
    result = ExtractorAgent().execute(state)
    assert all(f.extractor == "rules" for f in result.fields)


def test_model_values_must_quote_the_document():
    provider = EchoProvider(
        default='{"primary_diagnosis": {"value": "sepsis", "evidence": "never appears here"}}'
    )
    state = state_for("A note without a labelled diagnosis line.")
    result = ExtractorAgent(provider).execute(state)
    assert not any(f.name == "primary_diagnosis" for f in result.fields)
    assert any(i.code == "CF-EXT-012" for i in result.issues)


def test_model_values_with_a_real_quote_are_accepted():
    text = "Assessment: the patient has acute bronchitis and is improving."
    provider = EchoProvider(
        default='{"primary_diagnosis": {"value": "acute bronchitis", '
                '"evidence": "acute bronchitis"}}'
    )
    state = state_for(text)
    result = ExtractorAgent(provider).execute(state)
    field = next(f for f in result.fields if f.name == "primary_diagnosis")
    assert field.extractor == "model"
    assert text[field.span[0] : field.span[1]].lower() == "acute bronchitis"


def test_rule_confidence_always_beats_model_confidence():
    text = "Chief Complaint: chest pain"
    provider = EchoProvider(
        default='{"chief_complaint": {"value": "wrong", "evidence": "chest pain"}}'
    )
    state = state_for(text)
    state.add_result(ExtractorAgent(provider).execute(state))
    assert state.fields["chief_complaint"].value == "chest pain"
    assert state.fields["chief_complaint"].extractor == "rules"


# --- retriever ---------------------------------------------------------------
def _kb() -> KnowledgeBase:
    return KnowledgeBase(
        [
            Passage("P1", "Imaging Policy", "mri",
                    "Lumbar MRI requires six weeks of documented conservative therapy."),
            Passage("P2", "Anticoagulation", "warfarin",
                    "Warfarin interacts with aspirin and ibuprofen, increasing bleeding risk."),
            Passage("P3", "Discharge Standard", "elements",
                    "A discharge summary must record the discharge date and primary diagnosis."),
        ]
    )


def test_retriever_finds_the_relevant_passage():
    state = state_for("Requested Service: MRI lumbar spine", DocumentType.PRIOR_AUTHORIZATION)
    state.add_result(ExtractorAgent().execute(state))
    result = RetrieverAgent(knowledge_base=_kb()).execute(state)
    assert result.output["citations"][0].startswith("P1")


def test_retriever_reports_citations_for_every_hit():
    state = state_for("warfarin and aspirin together", DocumentType.CLINICAL_NOTE)
    result = RetrieverAgent(knowledge_base=_kb()).execute(state)
    assert all("#" in citation for citation in result.output["citations"])


def test_retriever_warns_when_nothing_matches():
    state = state_for("zzz qqq", DocumentType.UNKNOWN)
    result = RetrieverAgent(knowledge_base=_kb()).execute(state)
    assert any(i.code == "CF-RET-002" for i in result.issues)


def test_retriever_handles_an_empty_knowledge_base():
    result = RetrieverAgent(knowledge_base=KnowledgeBase()).execute(state_for("anything"))
    assert result.confidence == 0.0
    assert any(i.code == "CF-RET-001" for i in result.issues)


def test_bm25_prefers_the_more_specific_passage():
    hits = _kb().search("warfarin bleeding risk", top_k=3)
    assert hits[0].passage.doc_id == "P2"


def test_query_is_built_from_extracted_fields():
    state = state_for("Requested Service: MRI lumbar spine", DocumentType.PRIOR_AUTHORIZATION)
    state.add_result(ExtractorAgent().execute(state))
    query = RetrieverAgent(knowledge_base=_kb()).build_query(state)
    assert "MRI lumbar spine" in query


# --- validator ---------------------------------------------------------------
def _validated(text: str, doc_type: DocumentType = DocumentType.UNKNOWN) -> WorkflowState:
    state = state_for(text, doc_type)
    state.add_result(ExtractorAgent().execute(state))
    state.add_result(ValidatorAgent().execute(state))
    return state


def test_normal_values_produce_no_range_issues():
    state = _validated("HR 78, Temp 98.4 F, Potassium 4.1 mEq/L")
    assert not [i for i in state.issues if i.code.startswith("CF-VAL-00")]


def test_out_of_range_value_is_a_warning():
    state = _validated("Potassium 5.6 mEq/L")
    issue = next(i for i in state.issues if i.field_name == "potassium")
    assert issue.severity is Severity.WARNING


def test_critically_abnormal_value_is_critical_but_not_blocking():
    state = _validated("Potassium 6.4 mEq/L")
    issue = next(i for i in state.issues if i.field_name == "potassium")
    assert issue.severity is Severity.CRITICAL
    # It is a real finding that must reach a clinician, not a reason to
    # discard the document.
    assert issue.blocking is False


def test_impossible_value_is_an_extraction_error():
    state = _validated("Potassium 41 mEq/L")
    issue = next(i for i in state.issues if i.field_name == "potassium")
    assert issue.code == "CF-VAL-001"
    assert issue.severity is Severity.ERROR


def test_hypertensive_crisis_is_flagged():
    state = _validated("BP 196/124")
    assert any(i.code == "CF-VAL-012" for i in state.issues)


def test_inverted_blood_pressure_is_an_error():
    state = _validated("BP 78/122")
    assert any(i.code == "CF-VAL-011" for i in state.issues)


def test_missing_required_field_is_reported():
    state = _validated("Hospital Course: uneventful.", DocumentType.DISCHARGE_SUMMARY)
    issue = next(i for i in state.issues if i.code == "CF-VAL-020")
    assert "patient_name" in issue.detail["missing"]


def test_discharge_before_admission_is_an_error():
    text = "Admission Date: 03/06/2026\nDischarge Date: 03/02/2026\n"
    state = _validated(text)
    assert any(i.code == "CF-VAL-030" for i in state.issues)


def test_future_date_is_an_error():
    state = _validated("Admission Date: 01/01/2099")
    assert any(i.code == "CF-VAL-032" for i in state.issues)


def test_drug_interaction_is_detected():
    state = _validated("Medications: Warfarin 5 mg daily, Aspirin 81 mg daily")
    issue = next(i for i in state.issues if i.code == "CF-VAL-040")
    assert set(issue.detail["drugs"]) == {"warfarin", "aspirin"}
    assert issue.severity is Severity.CRITICAL


def test_implausible_dose_is_flagged():
    state = _validated("Medications: Warfarin 500 mg daily")
    assert any(i.code == "CF-VAL-041" for i in state.issues)


def test_medication_without_a_frequency_is_flagged():
    state = _validated("Medications: Lisinopril 10 mg")
    assert any(i.code == "CF-VAL-042" for i in state.issues)


def test_prescribing_a_documented_allergen_is_critical():
    state = _validated("Allergies: penicillin\nMedications: Penicillin 500 mg daily")
    issue = next(i for i in state.issues if i.code == "CF-VAL-043")
    assert issue.severity is Severity.CRITICAL


def test_validator_reports_document_validity_without_failing():
    state = state_for("Potassium 41 mEq/L")
    state.add_result(ExtractorAgent().execute(state))
    result = ValidatorAgent().execute(state)
    # Finding problems is the job; the agent itself succeeded.
    assert result.ok is True
    assert result.output["document_valid"] is False


@pytest.mark.parametrize(
    "value,expected", [("03/02/2026", 2026), ("2026-03-02", 2026), ("nonsense", None)]
)
def test_date_parsing(value, expected):
    parsed = parse_date(value)
    assert (parsed.year if parsed else None) == expected


# --- summarizer --------------------------------------------------------------
def test_template_summary_contains_only_extracted_values():
    state = _validated(DISCHARGE, DocumentType.DISCHARGE_SUMMARY)
    result = SummarizerAgent().execute(state)
    summary = result.output["summary"]
    assert "Maria Alvarez" in summary
    assert "Community acquired pneumonia" in summary
    assert result.output["source"] == "template"
    assert result.confidence == 1.0


def test_summary_lists_flagged_issues():
    state = _validated("Potassium 6.4 mEq/L")
    summary = SummarizerAgent().execute(state).output["summary"]
    assert "Flagged for review" in summary


def test_unsafe_model_summary_falls_back_to_the_template():
    state = _validated(DISCHARGE, DocumentType.DISCHARGE_SUMMARY)
    provider = EchoProvider(default="Call the patient on 415-555-0132 immediately.")
    result = SummarizerAgent(provider).execute(state)
    assert result.output["source"] == "template"
    assert any(i.code == "CF-SUM-003" for i in result.issues)
    assert "415-555-0132" not in result.output["summary"]


def test_safe_model_summary_is_used():
    state = _validated(DISCHARGE, DocumentType.DISCHARGE_SUMMARY)
    provider = EchoProvider(default="Patient recovered from pneumonia and was discharged.")
    result = SummarizerAgent(provider).execute(state)
    assert result.output["source"] == "model"
    assert result.confidence < 1.0


def test_lab_analytes_are_not_parsed_as_medications():
    # "Hemoglobin 13.1 g/dL" fits the drug name-dose-unit shape exactly.
    text = (
        "Labs: Hemoglobin 13.1 g/dL, Creatinine 0.9 mg/dL\n\n"
        "Discharge Medications: Lisinopril 10 mg PO daily\n"
    )
    state = state_for(text)
    state.add_result(ExtractorAgent().execute(state))
    drugs = {entry["drug"].lower() for entry in state.fields["medications"].value}
    assert drugs == {"lisinopril"}


def test_medication_spans_point_at_the_whole_document():
    text = "Header line\n\nMedications: Lisinopril 10 mg daily\n"
    state = state_for(text)
    result = ExtractorAgent().execute(state)
    field = next(f for f in result.fields if f.name == "medications")
    start, end = field.span
    assert "Lisinopril" in text[start:end]


def test_medications_found_without_a_section_header():
    state = state_for("Patient takes Lisinopril 10 mg daily for hypertension.")
    state.add_result(ExtractorAgent().execute(state))
    assert state.fields["medications"].value[0]["drug"].lower() == "lisinopril"
