# Handwriting Data Collection Protocol

## 1. Overview

This protocol outlines the procedures for collecting handwriting samples from students for the Dyslexia Early Detection System.

## 2. Equipment Requirements

### Preferred Method: Scanner
- Flatbed scanner with minimum 300 DPI
- Automatic document feeder (optional)
- TWAIN compatible

### Alternative Method: Photography
- Smartphone/tablet with 12+ MP camera
- Ring light or consistent lighting
- Copy stand or flat surface
- Whiteboard app or grid background

### Supplies
- Standard lined paper (various sizes)
- Blank unlined paper
- Pencil (#2) and eraser for students
- Timer/stopwatch

## 3. Paper Types and Tasks

### Task Type 1: Copying (Ages 5-8)
**Paper:** Primary lined (larger lines, dashed midline)
**Content:** 
```
The cat sat on the mat.
The dog ran in the park.
I like to read books.
```

### Task Type 2: Free Writing (Ages 8-12)
**Paper:** Standard lined
**Prompt:** "Write about your favorite day. Tell me what happened from start to finish."
**Time:** 10 minutes

### Task Type 3: Dictation (All ages)
**Paper:** Standard lined
**Content:** Age-appropriate passage (similar to speech passages)
**Process:** Teacher reads, students write

### Task Type 4: Essay (Ages 10-12)
**Paper:** Unlined or college-ruled
**Prompt:** "Describe a time when you learned something new. What did you learn and how did you feel?"
**Time:** 15-20 minutes

## 4. Collection Procedure

### Step 1: Preparation
- [ ] Verify parental consent obtained
- [ ] Gather appropriate paper type
- [ ] Prepare writing instrument
- [ ] Set timer if needed
- [ ] Label paper with student ID code

### Step 2: Administration
1. Explain task clearly to student
2. Demonstrate if needed
3. Start timer (if applicable)
4. Observe without helping
5. Collect completed sample

### Step 3: Digitization

**Scanning Method:**
- 300 DPI minimum
- Color: Grayscale
- Format: PNG or TIFF
- File size: < 20 MB
- Crop to writing area

**Photography Method:**
- Even lighting (no shadows)
- 90-degree angle to paper
- Include ruler for scale reference
- Multiple shots if needed
- Crop and straighten in software

### Step 4: Quality Check
- [ ] All writing captured
- [ ] No blur or motion
- [ ] Adequate contrast
- [ ] Correct orientation
- [ ] Re-scan if quality unacceptable

### Step 5: Storage
- Save to secure, encrypted storage
- File naming: `{student_id}_{task_type}_{date}.png`
- Log metadata in collection spreadsheet

## 5. Metadata to Record

| Field | Description | Example |
|-------|-------------|---------|
| student_id | Original ID (for linking) | S001 |
| task_type | Type of writing | copying |
| age | Student age | 7 |
| grade | Student grade | 2 |
| paper_type | Type of paper used | primary_lined |
| instrument | Pencil/pen used | pencil |
| time_taken | Minutes to complete | 5 |
| collection_date | Date collected | 2026-04-12 |
| digitization_method | How captured | scanner |
| quality | Sample quality | good |
| notes | Any observations | Student erased multiple times |

## 6. Quality Criteria

### Acceptable
- All text clearly visible
- Good contrast between ink and paper
- No folds or creases in writing area
- Proper orientation (not rotated)
- Complete task

### Unacceptable
- Significant blur
- Poor contrast
- Incomplete task
- Major rotation (> 15 degrees)
- Shadows obscuring text

## 7. Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Too dark | Adjust scanner brightness |
| Too light | Increase contrast or use threshold |
| Skewed image | Straighten in image editor |
| Large file size | Compress to PNG |
| Missing text | Re-scan entire sample |

## 8. Data Privacy

- All images stored with student code only
- Original IDs separated from images
- Encryption at rest enabled
- Access restricted to research team
- Deletion after retention period

---

*Protocol Version: 1.0*  
*Last Updated: 2026-04-12*