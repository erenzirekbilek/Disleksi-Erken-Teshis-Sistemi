<template>
  <div class="home">
    <header class="page-header">
      <h1>New Assessment</h1>
      <p>Enter student information and upload assessment materials</p>
    </header>

    <div class="layout">
      <div class="form-section">
        <div class="card form-card">
          <div class="card-header">
            <div class="card-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M16 7a4 4 0 1 1-8 0 4 4 0 0 1 8 0z"/>
                <path d="M8 21V21H4V3h16v18h-4"/>
              </svg>
            </div>
            <div>
              <h2>Student Information</h2>
              <p>Enter the student's unique identifier</p>
            </div>
          </div>
          
          <div class="form-group">
            <label class="form-label">Student ID</label>
            <input 
              v-model="form.studentId" 
              type="text" 
              class="form-input" 
              placeholder="e.g., STU-001"
              required
            />
          </div>
        </div>

        <div class="card form-card">
          <div class="card-header">
            <div class="card-icon speech">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                <line x1="12" y1="19" x2="12" y2="23"/>
                <line x1="8" y1="23" x2="16" y2="23"/>
              </svg>
            </div>
            <div>
              <h2>Speech Analysis</h2>
              <p>Audio recording for speech pattern analysis</p>
            </div>
          </div>
          
          <div 
            class="upload-zone"
            :class="{ 'has-file': audioFile, 'drag-over': audioDragOver }"
            @dragover.prevent="audioDragOver = true"
            @dragleave.prevent="audioDragOver = false"
            @drop.prevent="handleAudioDrop"
          >
            <div class="upload-content" v-if="!audioFile">
              <div class="upload-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                  <path d="M19 11a7 7 0 1 1-14 0"/>
                  <path d="M12 3v4M8 7l4-4 4 4"/>
                </svg>
              </div>
              <p class="upload-text">Drag & drop audio file or <label class="upload-link">browse<input type="file" accept="audio/*" class="hidden-input" @change="handleAudioUpload"/></label></p>
              <span class="upload-hint">WAV, MP3, M4A (max 10MB)</span>
            </div>
            <div class="file-preview" v-else>
              <div class="file-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                  <path d="M9 18V5l12-2v13"/>
                  <circle cx="6" cy="18" r="3"/>
                  <circle cx="18" cy="16" r="3"/>
                </svg>
              </div>
              <div class="file-info">
                <span class="file-name">{{ audioFile.name }}</span>
                <span class="file-size">{{ formatFileSize(audioFile.size) }}</span>
              </div>
              <button type="button" class="remove-btn" @click="audioFile = null">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <line x1="18" y1="6" x2="6" y2="18"/>
                  <line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
              </button>
            </div>
          </div>
        </div>

        <div class="card form-card">
          <div class="card-header">
            <div class="card-icon handwriting">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"/>
              </svg>
            </div>
            <div>
              <h2>Handwriting Analysis</h2>
              <p>Image of handwritten sample</p>
            </div>
          </div>
          
          <div 
            class="upload-zone"
            :class="{ 'has-file': imageFile, 'drag-over': imageDragOver }"
            @dragover.prevent="imageDragOver = true"
            @dragleave.prevent="imageDragOver = false"
            @drop.prevent="handleImageDrop"
          >
            <div class="upload-content" v-if="!imageFile">
              <div class="upload-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                  <circle cx="8.5" cy="8.5" r="1.5"/>
                  <polyline points="21 15 16 10 5 21"/>
                </svg>
              </div>
              <p class="upload-text">Drag & drop image or <label class="upload-link">browse<input type="file" accept="image/*" class="hidden-input" @change="handleImageUpload"/></label></p>
              <span class="upload-hint">PNG, JPG, JPEG (max 5MB)</span>
            </div>
            <div class="file-preview" v-else>
              <div class="file-icon image">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                  <circle cx="8.5" cy="8.5" r="1.5"/>
                  <polyline points="21 15 16 10 5 21"/>
                </svg>
              </div>
              <div class="file-info">
                <span class="file-name">{{ imageFile.name }}</span>
                <span class="file-size">{{ formatFileSize(imageFile.size) }}</span>
              </div>
              <button type="button" class="remove-btn" @click="imageFile = null">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <line x1="18" y1="6" x2="6" y2="18"/>
                  <line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
              </button>
            </div>
          </div>
        </div>

        <div class="card form-card">
          <div class="card-header">
            <div class="card-icon text">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
                <line x1="16" y1="13" x2="8" y2="13"/>
                <line x1="16" y1="17" x2="8" y2="17"/>
              </svg>
            </div>
            <div>
              <h2>Text Analysis</h2>
              <p>Text written by the student</p>
            </div>
          </div>
          
          <textarea 
            v-model="form.text"
            class="form-input text-input"
            placeholder="Enter the student's written text here..."
            rows="5"
          ></textarea>
        </div>

        <button 
          type="submit" 
          class="btn btn-primary submit-btn"
          :disabled="isLoading || !form.studentId"
          @click="submitAnalysis"
        >
          <span v-if="isLoading" class="loading-spinner"></span>
          <svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
            <polyline points="22 4 12 14.01 9 11.01"/>
          </svg>
          <span>{{ isLoading ? 'Analyzing...' : 'Run Assessment' }}</span>
        </button>
      </div>

      <div class="results-section" v-if="results || error">
        <div class="card results-card" v-if="results">
          <div class="results-header">
            <h2>Assessment Results</h2>
            <span class="timestamp">Completed {{ formatDate(results.timestamp) }}</span>
          </div>
          
          <div class="risk-indicator" :class="results.overall_risk">
            <div class="risk-circle" :class="results.overall_risk">
              <span class="risk-percent">{{ Math.round(results.overall_score * 100) }}%</span>
            </div>
            <div class="risk-info">
              <span class="risk-label">Overall Dyslexia Risk</span>
              <span class="risk-level" :class="results.overall_risk">{{ results.overall_risk.toUpperCase() }}</span>
            </div>
          </div>

          <div class="scores-grid">
            <div class="score-card">
              <div class="score-header">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                </svg>
                <span>Speech</span>
              </div>
              <div class="score-bar">
                <div class="score-fill" :style="{ width: `${results.speech_score * 100}%` }" :class="getRiskClass(results.speech_score)"></div>
              </div>
              <span class="score-value">{{ Math.round(results.speech_score * 100) }}%</span>
            </div>

            <div class="score-card">
              <div class="score-header">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"/>
                </svg>
                <span>Handwriting</span>
              </div>
              <div class="score-bar">
                <div class="score-fill" :style="{ width: `${results.handwriting_score * 100}%` }" :class="getRiskClass(results.handwriting_score)"></div>
              </div>
              <span class="score-value">{{ Math.round(results.handwriting_score * 100) }}%</span>
            </div>

            <div class="score-card">
              <div class="score-header">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                  <polyline points="14 2 14 8 20 8"/>
                </svg>
                <span>Text</span>
              </div>
              <div class="score-bar">
                <div class="score-fill" :style="{ width: `${results.text_score * 100}%` }" :class="getRiskClass(results.text_score)"></div>
              </div>
              <span class="score-value">{{ Math.round(results.text_score * 100) }}%</span>
            </div>
          </div>

          <div class="explanation-box" v-if="results.explanation">
            <h3>AI Analysis</h3>
            <p>{{ results.explanation }}</p>
          </div>

          <div class="features-box" v-if="results.top_features && results.top_features.length">
            <h3>Key Indicators</h3>
            <div class="features-list">
              <div v-for="modality in results.top_features" :key="modality.modality" class="feature-item">
                <span class="feature-modality">{{ modality.modality }}</span>
                <div class="feature-details">
                  <span v-for="(value, key) in modality.features" :key="key" class="feature-tag">
                    {{ formatKey(key) }}: {{ typeof value === 'number' ? value.toFixed(2) : value }}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="card error-card" v-if="error">
          <div class="error-content">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"/>
              <line x1="12" y1="8" x2="12" y2="12"/>
              <line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
            <p>{{ error }}</p>
          </div>
        </div>
      </div>

      <div class="results-section placeholder" v-else>
        <div class="placeholder-card">
          <div class="placeholder-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <path d="M9 5H7a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2"/>
              <rect x="9" y="3" width="6" height="4" rx="1"/>
              <path d="M9 12h6M9 16h6"/>
            </svg>
          </div>
          <h3>Results will appear here</h3>
          <p>Complete the assessment form and submit to see the analysis results</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { analysisService } from '../services/api'

export default {
  name: 'HomeView',
  data() {
    return {
      form: {
        studentId: '',
        text: ''
      },
      audioFile: null,
      imageFile: null,
      audioDragOver: false,
      imageDragOver: false,
      isLoading: false,
      results: null,
      error: null
    }
  },
  methods: {
    handleAudioUpload(event) {
      this.audioFile = event.target.files[0]
    },
    handleImageUpload(event) {
      this.imageFile = event.target.files[0]
    },
    handleAudioDrop(event) {
      this.audioDragOver = false
      const files = event.dataTransfer.files
      if (files.length && files[0].type.startsWith('audio/')) {
        this.audioFile = files[0]
      }
    },
    handleImageDrop(event) {
      this.imageDragOver = false
      const files = event.dataTransfer.files
      if (files.length && files[0].type.startsWith('image/')) {
        this.imageFile = files[0]
      }
    },
    formatFileSize(bytes) {
      if (bytes < 1024) return bytes + ' B'
      if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
      return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
    },
    formatDate(timestamp) {
      return new Date(timestamp).toLocaleString()
    },
    formatKey(key) {
      return key.replace(/_/g, ' ')
    },
    getRiskClass(score) {
      if (score >= 0.7) return 'high'
      if (score >= 0.4) return 'medium'
      return 'low'
    },
    async submitAnalysis() {
      this.isLoading = true
      this.error = null
      this.results = null

      try {
        this.results = await analysisService.analyze(
          this.form.studentId,
          this.audioFile,
          this.imageFile,
          this.form.text || null
        )
      } catch (err) {
        this.error = err.response?.data?.detail || 'Analysis failed. Please try again.'
      } finally {
        this.isLoading = false
      }
    }
  }
}
</script>

<style scoped>
.page-header {
  margin-bottom: 2rem;
}

.page-header h1 {
  font-size: 1.75rem;
  font-weight: 700;
  color: var(--gray-900);
  margin-bottom: 0.25rem;
}

.page-header p {
  color: var(--gray-500);
}

.layout {
  display: grid;
  grid-template-columns: 1fr 400px;
  gap: 1.5rem;
  align-items: start;
}

@media (max-width: 1024px) {
  .layout {
    grid-template-columns: 1fr;
  }
}

.form-section {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}

.form-card {
  padding: 1.5rem;
}

.card-header {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  margin-bottom: 1.25rem;
}

.card-icon {
  width: 40px;
  height: 40px;
  border-radius: 10px;
  background: var(--primary-light);
  color: var(--primary);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.card-icon svg {
  width: 20px;
  height: 20px;
}

.card-icon.speech {
  background: #f0f9ff;
  color: #0284c7;
}

.card-icon.handwriting {
  background: #fef3c7;
  color: #d97706;
}

.card-icon.text {
  background: #f3e8ff;
  color: #9333ea;
}

.card-header h2 {
  font-size: 1.125rem;
  font-weight: 600;
  color: var(--gray-800);
  margin-bottom: 0.125rem;
}

.card-header p {
  font-size: 0.875rem;
  color: var(--gray-500);
}

.upload-zone {
  border: 2px dashed var(--gray-300);
  border-radius: var(--radius-md);
  padding: 1.5rem;
  text-align: center;
  transition: all 0.2s;
  background: var(--gray-50);
}

.upload-zone:hover {
  border-color: var(--primary);
  background: var(--primary-light);
}

.upload-zone.drag-over {
  border-color: var(--primary);
  background: var(--primary-light);
  transform: scale(1.01);
}

.upload-zone.has-file {
  border-style: solid;
  border-color: var(--success);
  background: var(--success-light);
}

.upload-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
}

.upload-icon {
  width: 48px;
  height: 48px;
  background: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: var(--shadow-sm);
}

.upload-icon svg {
  width: 24px;
  height: 24px;
  color: var(--gray-400);
}

.upload-text {
  color: var(--gray-700);
  font-size: 0.9375rem;
}

.upload-link {
  color: var(--primary);
  cursor: pointer;
  font-weight: 500;
}

.upload-link:hover {
  text-decoration: underline;
}

.hidden-input {
  display: none;
}

.upload-hint {
  font-size: 0.8125rem;
  color: var(--gray-400);
}

.file-preview {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.file-icon {
  width: 40px;
  height: 40px;
  background: white;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.file-icon svg {
  width: 20px;
  height: 20px;
  color: var(--primary);
}

.file-icon.image {
  color: var(--success);
}

.file-info {
  flex: 1;
  text-align: left;
}

.file-name {
  display: block;
  font-weight: 500;
  color: var(--gray-800);
  font-size: 0.9375rem;
}

.file-size {
  font-size: 0.8125rem;
  color: var(--gray-500);
}

.remove-btn {
  width: 32px;
  height: 32px;
  border: none;
  background: white;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--gray-400);
  transition: all 0.2s;
}

.remove-btn:hover {
  background: var(--danger-light);
  color: var(--danger);
}

.remove-btn svg {
  width: 16px;
  height: 16px;
}

.text-input {
  resize: vertical;
  min-height: 120px;
  font-size: 0.9375rem;
  line-height: 1.6;
}

.submit-btn {
  width: 100%;
  padding: 1rem;
  font-size: 1rem;
  font-weight: 600;
  margin-top: 0.5rem;
}

.submit-btn svg {
  width: 20px;
  height: 20px;
}

.results-section {
  position: sticky;
  top: 100px;
}

.results-section.placeholder {
  display: flex;
}

.placeholder-card {
  background: white;
  border-radius: var(--radius-lg);
  padding: 2.5rem 1.5rem;
  text-align: center;
  border: 1px solid var(--gray-200);
  width: 100%;
}

.placeholder-icon {
  width: 64px;
  height: 64px;
  background: var(--gray-100);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 1rem;
}

.placeholder-icon svg {
  width: 32px;
  height: 32px;
  color: var(--gray-400);
}

.placeholder-card h3 {
  font-size: 1.125rem;
  color: var(--gray-700);
  margin-bottom: 0.5rem;
}

.placeholder-card p {
  font-size: 0.875rem;
  color: var(--gray-500);
  max-width: 240px;
  margin: 0 auto;
}

.results-card {
  padding: 1.5rem;
}

.results-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.results-header h2 {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--gray-800);
}

.timestamp {
  font-size: 0.8125rem;
  color: var(--gray-400);
}

.risk-indicator {
  display: flex;
  align-items: center;
  gap: 1.25rem;
  padding: 1.25rem;
  background: var(--gray-50);
  border-radius: var(--radius-md);
  margin-bottom: 1.5rem;
}

.risk-circle {
  width: 72px;
  height: 72px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 1.25rem;
}

.risk-circle.low {
  background: var(--success-light);
  color: var(--success);
}

.risk-circle.medium {
  background: var(--warning-light);
  color: var(--warning);
}

.risk-circle.high {
  background: var(--danger-light);
  color: var(--danger);
}

.risk-info {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.risk-label {
  font-size: 0.875rem;
  color: var(--gray-600);
}

.risk-level {
  font-weight: 600;
  font-size: 1rem;
}

.risk-level.low {
  color: var(--success);
}

.risk-level.medium {
  color: var(--warning);
}

.risk-level.high {
  color: var(--danger);
}

.scores-grid {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.score-card {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.875rem;
  background: var(--gray-50);
  border-radius: var(--radius-md);
}

.score-card .score-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  width: 120px;
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--gray-700);
}

.score-card .score-header svg {
  width: 16px;
  height: 16px;
}

.score-card .score-bar {
  flex: 1;
  height: 8px;
  background: var(--gray-200);
  border-radius: 4px;
  overflow: hidden;
}

.score-card .score-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.5s ease;
}

.score-card .score-fill.low {
  background: var(--success);
}

.score-card .score-fill.medium {
  background: var(--warning);
}

.score-card .score-fill.high {
  background: var(--danger);
}

.score-card .score-value {
  width: 48px;
  text-align: right;
  font-weight: 600;
  font-size: 0.875rem;
  color: var(--gray-700);
}

.explanation-box {
  padding: 1rem;
  background: #eff6ff;
  border-radius: var(--radius-md);
  border-left: 4px solid var(--primary);
  margin-bottom: 1.25rem;
}

.explanation-box h3 {
  font-size: 0.9375rem;
  font-weight: 600;
  color: var(--gray-800);
  margin-bottom: 0.5rem;
}

.explanation-box p {
  font-size: 0.9375rem;
  color: var(--gray-700);
  line-height: 1.7;
}

.features-box {
  padding: 1rem;
  background: var(--gray-50);
  border-radius: var(--radius-md);
}

.features-box h3 {
  font-size: 0.9375rem;
  font-weight: 600;
  color: var(--gray-800);
  margin-bottom: 0.75rem;
}

.features-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.feature-item {
  display: flex;
  flex-direction: column;
  gap: 0.375rem;
}

.feature-modality {
  font-size: 0.8125rem;
  font-weight: 600;
  text-transform: capitalize;
  color: var(--primary);
}

.feature-details {
  display: flex;
  flex-wrap: wrap;
  gap: 0.375rem;
}

.feature-tag {
  font-size: 0.75rem;
  padding: 0.25rem 0.5rem;
  background: white;
  border-radius: 4px;
  color: var(--gray-600);
}

.error-card {
  border: 1px solid var(--danger-light);
}

.error-content {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  color: var(--danger);
}

.error-content svg {
  width: 24px;
  height: 24px;
  flex-shrink: 0;
}

.error-content p {
  font-size: 0.9375rem;
}
</style>