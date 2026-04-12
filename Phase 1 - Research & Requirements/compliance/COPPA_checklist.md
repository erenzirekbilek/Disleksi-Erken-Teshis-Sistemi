# COPPA Compliance Checklist - Dyslexia Early Detection System

## Overview

This document outlines the compliance requirements for the Dyslexia Early Detection System under the Children's Online Privacy Protection Act (COPPA).

---

## 1. Applicability

### When COPPA Applies
- [x] Collecting personal information from children under 13
- [x] Commercial website or online service
- [x] Targeted to children
- [x] Actual knowledge of collecting from children

### Our Application
- [x] Target age: 5-12 (includes children under 13)
- [x] Educational context (school-based)
- [x] Parental consent required

---

## 2. Definitions

### Personal Information (PI)
Under COPPA, personal information includes:
- [x] First and last name
- [x] Home address
- [x] Online contact information
- [x] Screen name or user name
- [x] Telephone number
- [x] Social Security number

**For our system, we collect:**
- [x] Anonymized ID (not real name)
- [ ] Home address (NOT collected)
- [ ] Online contact (NOT collected)
- [x] Age/Grade (not PI)
- [x] Audio recordings (considered PI)
- [x] Handwriting images (considered PI)
- [x] Text samples (considered PI)

---

## 3. Requirements for Under-13 Users

### 3.1 Verifiable Parental Consent

**Methods Acceptable:**
| Method | Use Case | Implementation |
|--------|----------|----------------|
| Signed consent form | All cases | Paper form or digital signature |
| Email + consent verification | Low-risk | Email + follow-up |
| Credit card + nominal charge | High-risk | Not used |
| Government ID + consent | High-risk | Not used |
| Video conference | High-risk | Not used |

**Our Approach:**
- [x] Signed consent form (paper or digital)
- [x] Email verification with confirmation
- [x] Two-step verification recommended

### 3.2 Exceptions for School Consent

**School Exception (Section 99.31):**
- [x] Educational purpose
- [x] Direct benefit to educational process
- [x] School designates us as school official
- [x] School has control over data
- [x] Written agreement with school

**Our Implementation:**
- [x] Signed agreement with school district
- [x] School provides consent on behalf of parents
- [x] School verifies parental consent

---

## 4. Privacy Notice Requirements

### Required Elements for Children Under 13
- [x] Types of information collected
- [x] How information is collected
- [x] How information is used
- [x] Whether information is disclosed
- [x] Parental choices/controls
- [x] Contact information

### Platform Requirements
- [x] Direct link to privacy notice
- [x] Clear and understandable language
- [x] No hidden disclosures
- [x] Updated when practices change

---

## 5. Parental Rights

### Rights Afforded
- [x] Review child's information
- [x] Delete child's information
- [x] Refuse further collection
- [x] Revoke consent

### Implementation
- [x] Parent access portal
- [x] Deletion request process
- [x] Opt-out mechanism
- [x] Contact for concerns

---

## 6. Data Collection Limitations

### Collection Limitations
- [x] Collect only what's necessary
- [x] No collection for commercial purposes
- [x] No behavioral advertising
- [x] No requiring child to provide PI beyond what's necessary

### Our Data Collection
| Data Type | Collected | Purpose | Necessary |
|-----------|-----------|---------|-----------|
| Age/Grade | Yes | Age-appropriate prompts | Yes |
| Anonymized ID | Yes | Identify unique students | Yes |
| Audio | Yes | Speech analysis | Yes |
| Handwriting | Yes | Writing analysis | Yes |
| Text | Yes | Text analysis | Yes |

---

## 7. Security and Retention

### Security Requirements
- [x] Reasonable procedures to protect PI
- [x] Encryption for stored data
- [x] Secure transmission
- [x] Limited access to PI

### Retention
- [x] Retain only as long as necessary
- [x] Delete when purpose fulfilled
- [x] Secure deletion procedures
- [x] Defined retention schedule

**Our Retention Schedule:**
| Data Type | Retention | Reason |
|-----------|-----------|--------|
| Raw audio/images | 3 years | Quality verification |
| Processed features | 7 years | Longitudinal study |
| Analysis results | 7 years | Progress tracking |
| Consent forms | 7 years | Legal compliance |

---

## 8. Third-Party Services

### Third-Party Requirements
- [x] No third-party collection without consent
- [x] Contractual restrictions
- [x] No use beyond contract

### Our Third-Party Services
- [x] Cloud storage (encrypted)
- [x] ML model hosting
- [x] All under data processing agreements

---

## 9. School Consent Exception Requirements

### Requirements for School Consent (under COPPA)
- [x] Educational institution
- [x] For educational purpose
- [x] Parents notified
- [x] School designates contractor
- [x] Contract limits use

### Implementation Checklist
- [x] Signed school agreement
- [x] School verifies parent consent
- [x] School maintains control
- [x] We act as school official
- [x] Data used only for educational purpose
- [x] Parent can contact school

---

## 10. COPPA Implementation Checklist

### Pre-Collection
- [ ] Privacy notice published
- [ ] Parental consent form created
- [ ] School agreements signed
- [ ] Data processing agreements in place
- [ ] Security measures verified

### Collection Process
- [ ] Verifiable parental consent obtained
- [ ] Consent documented
- [ ] Data minimization followed
- [ ] No unauthorized collection

### Post-Collection
- [ ] Parent access available
- [ ] Deletion capability exists
- [ ] Retention policy followed
- [ ] Security maintained
- [ ] Data shared only as permitted

---

## 11. Sample Parental Consent Language

### Consent Form Must Include:
1. Name of website/operator
2. Types of information collected
3. How information is used
4. Whether information is disclosed
5. Parent's choices
6. Parent signature line
7. Date

### Required Disclosures:
- "We collect [specific types] from children under 13"
- "We use this information for [specific purpose]"
- "We do not [prohibited activities]"
- "You can review/delete information at any time"

---

## 12. Compliance Sign-Off

### Verification
| Requirement | Status | Date |
|-------------|--------|------|
| Privacy notice for under-13 | [ ] | |
| Parental consent mechanism | [ ] | |
| School consent documentation | [ ] | |
| Data security measures | [ ] | |
| Deletion capability | [ ] | |
| Access for parents | [ ] | |

### Signatures

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Project Lead | | | |
| Legal Advisor | | | |
| School Liaison | | | |

---

## 13. Key Differences: GDPR vs FERPA vs COPPA

| Aspect | GDPR | FERPA | COPPA |
|--------|------|-------|-------|
| Age | Under 16 (varies) | Under 18 | Under 13 |
| Consent | Parental consent | School consent | Verifiable parental consent |
| Rights | Comprehensive | Educational records | Online PI |
| Scope | Any processing | Educational records | Online services |

---

## 14. Summary: Our Compliance Approach

### For US Schools (COPPA + FERPA)
1. School signs agreement designating us as "school official"
2. School obtains parental consent (or uses school exception)
3. School verifies consent before data collection
4. Parent contacts school for access/deletion

### For EU Schools (GDPR)
1. Parental consent obtained directly
2. Age-appropriate data handling
3. Full data subject rights
4. DPO notification if required

---

*Last Updated: [Date]*  
*Version: 1.0*  
*Note: Consult legal counsel for specific implementation requirements.*