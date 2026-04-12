# Phase 4: Model Development

**Duration:** 4-6 weeks  
**Start:** Week 12  
**End:** Week 17  
**Dependencies:** Phase 3 Complete  
**Key Stakeholders:** ML Research Lead, Model Training Engineer

---

## 4.1 Overview

Phase 4 focuses on training and validating modality-specific models, then building the fusion layer for combined risk scoring.

**Goals:**
- Train speech model (CNN/LSTM/Transformer)
- Train handwriting model (ResNet/EfficientNet/ViT)
- Train text model (BERT/DistilBERT)
- Build fusion layer for combined scoring

---

## 4.2 Week 12-14: Speech Model Development

### Task 4.1.1 — Design Speech Model Architecture

**Primary:** CNN on spectrogram input  
**Alternative:** RNN/LSTM for temporal sequences  
**Alternative:** Transformer-based audio model

### Task 4.1.2 — Train and Evaluate Speech Model

- Data split: 70% train, 15% validation, 15% test (stratified)
- Hyperparameter search
- Output: Speech risk score `S_speech`

---

## 4.3 Week 12-14: Handwriting Model Development

### Task 4.2.1 — Design Handwriting Model Architecture

**Primary:** EfficientNet-B0 (transfer learning)  
**Alternative:** ResNet-34, Vision Transformer (ViT)

### Task 4.2.2 — Train and Evaluate Handwriting Model

- Data augmentation: rotation, scale, flip, brightness
- Transfer learning from ImageNet
- Output: Writing risk score `S_writing`

---

## 4.4 Week 12-14: Text Model Development

### Task 4.3.1 — Design Text Model Architecture

**Primary:** BERT-base or DistilBERT  
**Fine-tuning:** 3-class classification (Low/Medium/High)

### Task 4.3.2 — Train and Evaluate Text Model

- Learning rate: 2e-5 to 5e-5
- Epochs: 3-5 with early stopping
- Output: NLP risk score `S_nlp`

---

## 4.5 Weeks 15-17: Fusion Layer Development

### Task 4.4.1 — Implement Baseline Fusion

**Weighted Sum:** `R = w1·S_speech + w2·S_writing + w3·S_nlp`

### Task 4.4.2 — Implement Advanced Fusion

**Recommended:** Ensemble meta-model  
- Gradient Boosting (XGBoost/LightGBM) or MLP
- Takes all three scores as input

### Task 4.4.3 — Cross-Validation

- k-fold cross-validation (k=5 or k=10)

---

## 4.6 Deliverables Summary

| Deliverable | Description | Owner | Due |
|-------------|-------------|-------|-----|
| Trained Speech Model | Production-ready model with S_speech output | ML Engineer | Week 14 |
| Trained Handwriting Model | Production-ready model with S_writing output | ML Engineer | Week 14 |
| Trained Text Model | Production-ready model with S_nlp output | ML Engineer | Week 14 |
| Fusion Model | Combined risk scoring engine (R) | ML Engineer | Week 17 |
| Model Performance Reports | Full metrics for all models | ML Engineer | Week 17 |
| Cross-Validation Results | k-fold results (k=5 or 10) | Data Scientist | Week 17 |

---

## 4.7 Key Milestones

| Milestone | Description | Success Criteria |
|-----------|-------------|------------------|
| M4.1 | Speech model trained | F1 > 0.80 on validation |
| M4.2 | Handwriting model trained | F1 > 0.80 on validation |
| M4.3 | Text model trained | F1 > 0.80 on validation |
| M4.4 | Fusion model trained | F1 > 0.85 on validation |
| M4.5 | Cross-validation complete | All models validated |

---

## 4.8 Phase 4 Completion Checklist

- [ ] Speech model architecture designed
- [ ] Speech model trained and evaluated
- [ ] Handwriting model architecture designed
- [ ] Handwriting model trained and evaluated
- [ ] Text model architecture designed
- [ ] Text model trained and evaluated
- [ ] Baseline fusion implemented
- [ ] Advanced fusion implemented
- [ ] Cross-validation completed

---

*Next Phase: Phase 5 — Explainability (XAI) Layer*