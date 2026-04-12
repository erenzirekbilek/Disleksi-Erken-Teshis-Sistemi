# Phase 1: Research & Requirements

**Duration:** 2 weeks  
**Start:** Week 1  
**End:** Week 2  
**Dependencies:** None (Foundation Phase)  
**Key Stakeholders:** Project Lead, Clinical Advisor, Data Protection Officer

---

## 1.1 Overview

Phase 1 establishes the scientific and technical foundation before any development begins. This phase ensures the project is clinically grounded, technically sound, and legally compliant.

**Goals:**
- Establish clinical and scientific basis for dyslexia detection
- Define clear requirements and user personas
- Ensure regulatory compliance (GDPR, FERPA, COPPA)

---

## 1.2 Week 1: Literature Review and Clinical Foundation

### Task 1.1.1 — Comprehensive Review of Clinical Dyslexia Indicators

**Description:** Conduct in-depth research on clinical dyslexia markers and indicators.

**Activities:**
- Review phonological processing deficit literature
- Document letter reversal patterns (b/d, p/q confusion)
- Compile reading fluency metrics and markers
- Gather 20+ peer-reviewed studies from speech-language pathology journals

**Deliverables:**
- Annotated literature review document (20+ key studies)
- Summary of dyslexia indicator taxonomy

**Owner:** Research Team  
**Due:** Week 1

---

### Task 1.1.2 — Document Dyslexic Handwriting Characteristic Markers

**Description:** Research and document handwriting patterns associated with dyslexia.

**Activities:**
- Letter size inconsistencies
- Spatial arrangement problems
- Baseline adherence issues
- Stroke characteristics
- Common reversal patterns

**Deliverables:**
- Handwriting marker documentation

**Owner:** Research Team  
**Due:** Week 1

---

### Task 1.1.3 — Compile Text-Based Dyslexia Indicators

**Description:** Research text and writing indicators of dyslexia.

**Activities:**
- Spelling error pattern frequencies
- Sentence structure abnormalities
- Reading comprehension metric correlations
- Writing fluency markers

**Deliverables:**
- Text-based indicator taxonomy

**Owner:** Research Team  
**Due:** Week 1

---

## 1.3 Week 1-2: Requirements Definition

### Task 1.2.1 — Define Target Age Group and Rationale

**Description:** Establish the target demographic for the detection system.

**Recommended Specifications:**
- **Age Range:** Ages 5-12 (critical intervention window)
- **Rationale:** Early elementary through middle elementary years
- **Age-specific assessment criteria** for each year

**Deliverables:**
- Target age group definition document

**Owner:** Product Manager  
**Due:** Week 2

---

### Task 1.2.2 — Create Detailed User Personas

**Description:** Define key user types and their needs.

| Persona | Role | Needs |
|---------|------|-------|
| **Teacher** | Classroom teacher (20-30 students) | Quick risk screening, class-level views, bulk analysis |
| **Parent** | Concerned parent | Clear explanations, single-child view, actionable insights |
| **School Psychologist** | Clinical assessor | Detailed metrics, evidence-based analysis, exportable reports |

**Deliverables:**
- User personas document with use cases
- User journey maps

**Owner:** UX Researcher  
**Due:** Week 2

---

### Task 1.2.3 — Establish Risk Classification Criteria

**Description:** Define the risk scoring framework.

**Risk Classification:**

| Risk Level | Score Range | Description |
|------------|-------------|-------------|
| **Low Risk** | 0.0 - 0.33 | Minimal indicators detected |
| **Medium Risk** | 0.34 - 0.66 | Some indicators present |
| **High Risk** | 0.67 - 1.0 | Strong indicators present |

**Deliverables:**
- Risk classification framework document

**Owner:** Clinical Advisor  
**Due:** Week 2

---

### Task 1.2.4 — Define Functional Requirements

**Description:** Document all functional requirements for the system.

**Core Requirements:**
1. **Speech Analysis**
   - Audio recording upload (WAV/MP3)
   - Phonological processing assessment
   - Fluency metrics calculation
   - Speech risk score output

2. **Handwriting Analysis**
   - Image upload (PNG/JPEG/TIFF)
   - Letter reversal detection
   - Spacing irregularity assessment
   - Handwriting risk score output

3. **Text Analysis**
   - Plain text input
   - Spelling error detection
   - Grammar assessment
   - Complexity scoring
   - Text risk score output

4. **Fusion & Reporting**
   - Combined risk score calculation
   - Risk class assignment (Low/Medium/High)
   - Plain-language explanations
   - Report generation

**Deliverables:**
- Functional requirements specification
- Use case documentation

**Owner:** Product Manager  
**Due:** Week 2

---

### Task 1.2.5 — Define Non-Functional Requirements

**Description:** Document performance, scalability, and quality requirements.

**Performance Requirements:**
- API response time < 5 seconds (per modality)
- Support for 100+ concurrent users
- File size limits: Audio 50MB, Images 20MB, Text 10,000 chars

**Scalability Requirements:**
- Horizontal scaling capability
- Microservices architecture
- Container-based deployment

**Quality Requirements:**
- Model accuracy target: F1 > 0.80 per modality
- End-to-end fusion accuracy: F1 > 0.85
- Test coverage: > 80%

**Deliverables:**
- Non-functional requirements document

**Owner:** Technical Lead  
**Due:** Week 2

---

## 1.4 Week 2: Compliance Framework

### Task 1.3.1 — Map Regulatory Requirements

**Description:** Identify and document all applicable regulations.

**Regulations:**

| Regulation | Region | Key Requirements |
|------------|--------|------------------|
| **GDPR** | EU | Data minimization, explicit consent, right to erasure, data portability |
| **FERPA** | US | Educational records protection, parental access rights, directory restrictions |
| **COPPA** | US | Additional protections for minors under 13, parental consent required |

**Deliverables:**
- Regulatory compliance mapping document

**Owner:** Legal/Compliance  
**Due:** Week 2

---

### Task 1.3.2 — Create Data Handling Protocols

**Description:** Establish protocols for ethical data collection and handling.

**Protocols Required:**
- Consent form templates for parents/guardians
- Data retention schedules (recommend 7 years post-graduation)
- Deletion procedures for data withdrawal
- Data access request procedures

**Deliverables:**
- Data handling protocols document
- Consent form templates

**Owner:** Ethics Committee  
**Due:** Week 2

---

### Task 1.3.3 — Define Evaluation Metrics

**Description:** Establish metrics for model and system evaluation.

**Model Metrics:**
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

**System Metrics:**
- Response time
- Uptime
- Concurrent users
- Error rate

**Deliverables:**
- Evaluation metrics specification

**Owner:** ML Research Lead  
**Due:** Week 2

---

## 1.5 Deliverables Summary

| Deliverable | Description | Owner | Due |
|-------------|-------------|-------|-----|
| Literature Review Document | Annotated collection of 20+ key studies | Research Team | Week 1 |
| Requirements Specification | Detailed functional and non-functional requirements | Product Manager | Week 2 |
| User Personas Document | 3 persona profiles with use cases | UX Researcher | Week 2 |
| Risk Classification Framework | Documented criteria for Low/Medium/High | Clinical Advisor | Week 2 |
| Regulatory Compliance Map | Requirements checklist for GDPR/FERPA/COPPA | Legal/Compliance | Week 2 |
| Data Collection Protocol | Ethical guidelines for data gathering | Ethics Committee | Week 2 |
| Evaluation Metrics Spec | Model and system evaluation criteria | ML Research Lead | Week 2 |

---

## 1.6 Resources Required

| Resource | Quantity | Role |
|----------|----------|------|
| Clinical Advisor | 1 FTE | Dyslexia expert for clinical guidance |
| Research Analyst | 1 FTE | Literature review and documentation |
| Data Protection Officer | 0.25 FTE | Compliance oversight |
| Product Manager | 0.5 FTE | Requirements synthesis |
| UX Researcher | 0.25 FTE | Persona development |
| Legal/Compliance | 0.25 FTE | Regulatory mapping |

---

## 1.7 Key Milestones

| Milestone | Description | Success Criteria |
|-----------|-------------|------------------|
| M1.1 | Literature review complete | Minimum 15 relevant studies analyzed |
| M1.2 | Requirements approved | Stakeholder sign-off received |
| M1.3 | Compliance framework validated | Legal counsel approval obtained |
| M1.4 | Risk classification defined | Clinical advisor sign-off |
| M1.5 | User personas approved | All stakeholders认同 |

---

## 1.8 Dependencies

- **External Dependencies:**
  - Clinical advisor availability for consultation
  - Access to dyslexia research databases

- **Internal Dependencies:**
  - None (this is the foundation phase)

---

## 1.9 Risks and Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Limited clinical expertise access | High | Engage dyslexia specialists early; consider advisory board |
| Ambiguous regulatory requirements | Medium | Consult with legal experts; document decisions |
| Scope creep from requirements | Medium | Strict change control; phase-gated approvals |
| Insufficient research data | Medium | Plan for multiple data sources; consider partnerships |

---

## 1.10 Phase 1 Completion Checklist

- [ ] Literature review completed (20+ studies)
- [ ] Target age group defined
- [ ] User personas created and approved
- [ ] Risk classification framework defined
- [ ] Functional requirements documented
- [ ] Non-functional requirements documented
- [ ] GDPR compliance mapped
- [ ] FERPA compliance mapped
- [ ] COPPA compliance mapped
- [ ] Data handling protocols created
- [ ] Evaluation metrics defined
- [ ] All stakeholders sign-off received

---

**Phase 1 Approval:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Project Lead | | | |
| Clinical Advisor | | | |
| Product Manager | | | |
| Data Protection Officer | | | |

---

*Next Phase: Phase 2 — Data Collection & Preparation*