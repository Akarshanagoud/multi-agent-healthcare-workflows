# Verified results

Output captured by actually running this repository. Nothing here is hand-written.

| | |
|---|---|
| Date | 2026-10-15 |
| Commit | `f6a67c9` |
| Python | Python 3.11.9 |
| OS | Windows 11 |

Timing figures (ms, rps) vary by machine and run. Counts and accuracy figures are deterministic.

## Reproduce

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]" pytest
pytest -q
python examples\quickstart.py
```

## Test suite

```text
........................................................................ [ 57%]
......................................................                   [100%]
126 passed in 0.68s
```

## `examples/quickstart.py`

```text

==========================================================================
1. PHI is minimised before any agent, let alone any model, sees the text
==========================================================================
removed: DOB, MRN
Patient: Maria Alvarez
MRN: <MRN_7AF1123C>
  (MRN is pseudonymised, not deleted: agents still need to tell patients apart)

==========================================================================
2. A clean document goes straight through
==========================================================================
decision   : complete
type       : discharge_summary
fields     : 15  mean confidence 0.87
duration   : 3.0 ms
agent trace:
  intake         0.00 ms  conf=1.00  fields=0
  router         0.20 ms  conf=1.00  fields=0
  retriever      0.12 ms  conf=0.70  fields=0
  extractor      0.17 ms  conf=0.87  fields=15
  validator      2.07 ms  conf=1.00  fields=0
  summarizer     0.18 ms  conf=1.00  fields=0

==========================================================================
3. Every field points at the text that produced it
==========================================================================
  patient_name         = Maria Alvarez
                         span 28-41, source text: 'Maria Alvarez'
  primary_diagnosis    = Community acquired pneumonia, right lower lobe
                         span 147-193, source text: 'Community acquired pneumonia, right lower lobe'
  blood_pressure       = 122/78
                         span 198-204, source text: '122/78'
  medications          = [{'drug': 'Amoxicillin', 'dose': 500.0, 'unit': 
                         span 323-375, source text: 'Amoxicillin 500 mg PO tid, Lisinopril 10 mg PO d'

==========================================================================
4. Clinical problems escalate to a human, they are not silently accepted
==========================================================================
decision: needs_review
  - heart_rate of 112.0 bpm is outside the normal range (60-100)
  - interaction: aspirin + warfarin - additive bleeding risk
  - blood pressure 168/94 is elevated

==========================================================================
5. An injected instruction is quarantined before any agent runs
==========================================================================
decision   : rejected
agents run : ['intake']
fields     : 0  (nothing was extracted from it)
  CF-INJ-001 blocking=True document contains text addressed to the model, not the reader
  CF-INJ-002 blocking=True document contains text addressed to the model, not the reader

==========================================================================
6. Severity and blocking are different questions
==========================================================================
A potassium of 6.4 is CRITICAL and urgent, but correctly extracted:
  decision  : needs_review  (not rejected)
  potassium : 6.4 - preserved for the clinician
An injected instruction is CRITICAL and untrustworthy:
  decision  : rejected  (rejected, nothing preserved)

==========================================================================
7. A fabricated extraction cannot reach a clinician
==========================================================================
The model claimed a diagnosis quoting text that is not in the document.
  extracted diagnosis: None
  CF-EXT-012 model proposed primary_diagnosis with a quote absent from the document; value discarded

==========================================================================
8. The summary is assembled from verified fields only
==========================================================================
Discharge Summary (DOC-A)

Patient: Maria Alvarez
DOB: <DOB>
MRN: <MRN_7AF1123C>
Diagnosis: Community acquired pneumonia, right lower lobe
Admitted: 03/02/2026
Discharged: 03/06/2026

Vitals: BP 122/78 mmHg, HR 78 bpm, Temp 98.4 F, SpO2 97 %

Labs: Hgb 13.1 g/dL, Cr 0.9 mg/dL, K 4.1 mEq/L

Medications:
  - Amoxicillin 500.0mg tid
  - Lisinopril 10.0mg daily

Allergies: none documented

Policy context: POL-DC-001#required-elements, GDL-RENAL-001#creatinine, GDL-LAB-001#notification, POL-PA-001#medical-necessity

==========================================================================
9. Batch processing produces a review queue, not just a number
==========================================================================
documents        : 10
straight-through : 20%
needs review     : 70%
mean duration    : 0.8 ms/document

review queue, worst first:
  DOC-005    rejected      critical  document contains embedded instructions
  DOC-002    needs_review  critical  hemoglobin of 9.4 g/dL is outside the normal range (12.0-17.
  DOC-003    needs_review  critical  heart_rate of 112.0 bpm is outside the normal range (60-100)
  DOC-009    needs_review  critical  blood pressure 196/124 is in the hypertensive crisis range
  DOC-007    needs_review  error     no fields could be extracted from the document
  DOC-010    needs_review  error     low routing confidence (0.00)
  DOC-006    needs_review  error     potassium of 41.0 mEq/L is physiologically impossible - like
  DOC-008    needs_review  warning   hba1c of 6.8 % is outside the normal range (4.0-5.7)

==========================================================================
10. Governance settings are configuration, not code
==========================================================================
default policy : complete
strict policy  : needs_review

Done. Every document above was processed with no model call required.
```
