# Text Data Collection Protocol

## 1. Overview

This protocol outlines the procedures for collecting text samples from students for the Dyslexia Early Detection System.

## 2. Collection Methods

### Method 1: Digital Submission (Preferred)
- Online form/Google Form
- Typed directly by student
- Auto-save functionality

### Method 2: Transcription
- Student writes by hand
- Research assistant transcribes
- Double-check transcription accuracy

### Method 3: Dictation
- Student responds verbally
- Audio recorded and transcribed
- Used for younger students

## 3. Prompts by Age Group

### Ages 5-7 (Kindergarten - 1st Grade)

**Prompt 1: Sentence Writing**
```
Write 3-5 sentences about what you did today.
Example: Today I played with my friend. We played in the park.
```

**Prompt 2: Picture Description**
```
Look at the picture. Write about what you see.
[Include simple illustration prompt]
```

**Prompt 3: Simple Story**
```
Write a story about a cat. Try to write 3-5 sentences.
```

### Ages 8-10 (2nd - 4th Grade)

**Prompt 1: Personal Narrative**
```
Write about your favorite memory. Tell me what happened, who was there, and how you felt.
(5-8 sentences)
```

**Prompt 2: Informational**
```
Explain how to make a sandwich. Tell me all the steps in order.
(5-8 sentences)
```

**Prompt 3: Creative Story**
```
You found a magic key. What happens next? Write a story about what you discover.
(8-12 sentences)
```

### Ages 11-12 (5th - 6th Grade)

**Prompt 1: Argumentative**
```
Should students have homework every night? Write to convince your reader.
(1-2 paragraphs)
```

**Prompt 2: Research Summary**
```
Choose a topic you know about. Write 2-3 paragraphs explaining what you know.
```

**Prompt 3: Narrative**
```
Write about a time when you faced a challenge. How did you handle it? What did you learn?
(2-3 paragraphs)
```

## 4. Collection Procedure

### Step 1: Preparation
- [ ] Verify parental consent obtained
- [ ] Select appropriate prompt by age
- [ ] Prepare digital form or paper
- [ ] Set timer if timed writing
- [ ] Prepare example if needed

### Step 2: Administration
1. Read prompt clearly to student
2. Allow time to think (1-2 minutes)
3. Start timer (if applicable)
4. Allow student to work independently
5. Do not provide spelling help
6. Collect when complete or time expires

### Step 3: Processing (Digital)
- Save as plain text (.txt)
- Verify no formatting issues
- Clean up any system artifacts
- File naming: `{student_id}_{prompt_id}_{date}.txt`

### Step 4: Processing (Handwritten)
- Transcribe within 24 hours
- Double-check against original
- Note any unclear words
- Save as plain text

### Step 5: Quality Check
- [ ] Complete response to prompt
- [ ] No system errors or artifacts
- [ ] Reasonable length (not too short)
- [ ] Original writing preserved for transcription

### Step 6: Storage
- Save to secure, encrypted storage
- Log metadata in collection spreadsheet

## 5. Metadata to Record

| Field | Description | Example |
|-------|-------------|---------|
| student_id | Original ID (for linking) | S001 |
| prompt_id | Prompt used | P1, P2, etc. |
| age | Student age | 9 |
| grade | Student grade | 3 |
| collection_method | How collected | digital |
| word_count | Words in response | 85 |
| time_taken | Minutes to complete | 12 |
| collection_date | Date collected | 2026-04-12 |
| transcriber | If transcribed | J. Smith |
| quality | Sample quality | good |
| notes | Any observations | Student asked for clarification |

## 6. Quality Criteria

### Acceptable
- Responds to prompt
- Complete sentences
- Reasonable length for age
- No system errors

### Unacceptable
- Does not address prompt
- Too short (< 20 words for older students)
- System errors/corruption
- Copied from another source

## 7. Special Considerations

### For Students with Motor Difficulties
- Offer typing option
- Allow extended time
- Accept shorter responses
- Do not penalize for length

### For English Language Learners
- Accept native language responses
- Note language in metadata
- Consider separate analysis
- Allow translation assistance

### For Students with ADHD
- Reduce distraction environment
- Allow movement breaks
- Offer shorter prompts if needed
- Allow dictate option

## 8. Data Privacy

- All text stored with student code only
- Original IDs separated from text files
- Encryption at rest enabled
- Access restricted to research team
- Deletion after retention period

---

*Protocol Version: 1.0*  
*Last Updated: 2026-04-12*