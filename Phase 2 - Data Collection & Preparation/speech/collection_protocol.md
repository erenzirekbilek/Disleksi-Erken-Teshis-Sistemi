# Speech Data Collection Protocol

## 1. Overview

This protocol outlines the procedures for collecting speech data from students for the Dyslexia Early Detection System.

## 2. Equipment Requirements

### Hardware
- Professional USB microphone (e.g., Blue Yeti, Audio-Technica AT2020)
- Pop filter
- Acoustic foam panels or quiet room
- Laptop/tablet for recording
- Headphones for monitoring

### Software
- Audio recording software (Audacity, GarageBand, or custom app)
- File naming utility

## 3. Environment Standards

| Parameter | Requirement |
|-----------|--------------|
| Background Noise | < 40 dB (quiet office/room) |
| Room Acoustics | Minimal echo |
| Distance to Mic | 6-12 inches |
| Position | Slight angle to avoid plosives |

## 4. Recording Setup

### Equipment Configuration
- Sample Rate: 48 kHz
- Bit Depth: 16-bit
- Channels: Mono
- Format: WAV (lossless)

### Test Recording
1. Have student say their name
2. Play back to verify clarity
3. Check for clipping or distortion
4. Adjust microphone position if needed

## 5. Reading Passages

### Passage A (Easy - Ages 5-7)
```
The cat sat on the mat. 
It is a big red cat.
The cat likes to play.
```

### Passage B (Medium - Ages 8-10)
```
Once upon a time, there was a little boy who lived in a small village.
Every day, he would walk to the river to fetch water for his family.
One day, he found a magical key shining in the sand.
```

### Passage C (Hard - Ages 11-12)
```
The scientific method has revolutionized our understanding of the natural world.
Through careful observation, hypothesis formation, and rigorous experimentation,
researchers have been able to uncover the fundamental laws that govern the universe.
```

## 6. Collection Procedure

### Step 1: Preparation
- [ ] Verify parental consent obtained
- [ ] Select appropriate passage by age
- [ ] Test audio equipment
- [ ] Create file naming: `{student_id}_{passage_id}_{date}.wav`

### Step 2: Recording
1. Greet student and explain task
2. Have student practice reading passage silently
3. Start recording
4. Allow student to read at natural pace
5. Stop recording after completion
6. Thank student

### Step 3: Quality Check
- [ ] Check for background noise
- [ ] Verify clear audio throughout
- [ ] Confirm student completed entire passage
- [ ] Re-record if quality unacceptable

### Step 4: Storage
- Save to secure, encrypted storage
- Do not modify original files
- Log metadata in collection spreadsheet

## 7. Metadata to Record

| Field | Description | Example |
|-------|-------------|---------|
| student_id | Original ID (for linking) | S001 |
| passage_id | Passage used | A, B, or C |
| age | Student age | 7 |
| recording_date | Date collected | 2026-04-12 |
| duration_seconds | Recording length | 45 |
| quality | Audio quality rating | good |
| notes | Any observations | Student hesitant at start |

## 8. Quality Criteria

### Acceptable
- Clear speech throughout
- No background conversation/music
- Complete passage read
- No technical glitches

### Unacceptable
- Significant background noise
- Incomplete passage
- Audio clipping
- Distorted recording

## 9. Data Privacy

- All recordings stored with student ID only
- Original IDs separated from audio files
- Encryption at rest enabled
- Access restricted to research team
- Deletion after retention period

---

*Protocol Version: 1.0*  
*Last Updated: 2026-04-12*