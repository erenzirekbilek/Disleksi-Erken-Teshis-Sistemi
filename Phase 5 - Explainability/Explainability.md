# Phase 5: Explainability (XAI) Layer

**Duration:** 1-2 weeks  
**Start:** Week 18  
**End:** Week 19  
**Dependencies:** Phase 4 Complete  
**Key Stakeholders:** ML Engineer, XAI Specialist, UX Writer

---

## 5.1 Overview

Phase 5 focuses on making the system's decisions transparent and understandable to non-technical users through explainable AI techniques.

**Goals:**
- Implement SHAP for global and local feature importance
- Implement LIME for per-sample explanations
- Integrate LLM for plain-language explanations

---

## 5.2 Week 18: SHAP and LIME Integration

### Task 5.1.1 — Implement SHAP for Speech Model

- KernelSHAP for model-agnostic explanations
- TreeSHAP for tree-based models
- Global feature importance visualization
- Local explanations per prediction

### Task 5.1.2 — Implement SHAP for Handwriting Model

- Per-pixel importance maps
- Feature importance overlay on images
- Error pattern explanations

### Task 5.1.3 — Implement SHAP for Text Model

- Token-level importance
- Sentence-level explanations
- Word-level attribution visualization

### Task 5.2.1 — Implement LIME Explanations

- Local interpretable predictions
- Per-modality adapters
- Text: word masking, Image: superpixel masking, Audio: segment masking

---

## 5.3 Week 19: LLM Explanation Generation

### Task 5.3.1 — Design Prompt Templates

- Input: Model outputs + top features + risk level
- Output: Plain language explanations (50-100 words)
- Template examples for each risk level

### Task 5.3.2 — Implement LLM Integration

- API integration (OpenAI GPT, Anthropic Claude, or open-source)
- Request batching for cost optimization
- Caching strategy

### Task 5.3.3 — Create Explanation Templates

- Phonological difficulty explanation
- Handwriting difficulty explanation
- Text difficulty explanation
- Combined modality explanations

### Task 5.4.1 — Expert Review Cycle

- Clinical advisor verification
- Educator feedback (5 teachers minimum)
- Iterative template improvements

---

## 5.4 Deliverables Summary

| Deliverable | Description | Owner | Due |
|-------------|-------------|-------|-----|
| SHAP Integration | All three modalities | ML Engineer | Week 18 |
| LIME Integration | Per-modality adapters | ML Engineer | Week 18 |
| LLM Explanation Pipeline | Automated plain language outputs | ML Engineer | Week 19 |
| Explanation Quality Report | Expert review summary | Clinical Advisor | Week 19 |

---

## 5.5 Key Milestones

| Milestone | Description | Success Criteria |
|-----------|-------------|------------------|
| M5.1 | SHAP operational | All modalities covered |
| M5.2 | LIME operational | Local explanations working |
| M5.3 | LLM explanations generated | Quality feedback positive |

---

## 5.6 Phase 5 Completion Checklist

- [ ] SHAP integrated for speech model
- [ ] SHAP integrated for handwriting model
- [ ] SHAP integrated for text model
- [ ] LIME integrated for all modalities
- [ ] LLM prompt templates designed
- [ ] LLM integration implemented
- [ ] Explanation quality review completed

---

*Next Phase: Phase 6 — Backend & AI Microservices*