<template>
  <div class="home">
    <div class="hero">
      <h1>Dyslexia Early Detection System</h1>
      <p>Multi-modal AI-powered screening for early dyslexia risk assessment</p>
    </div>

    <div class="content-grid">
      <div class="analysis-card card">
        <h2>New Assessment</h2>
        
        <form @submit.prevent="submitAnalysis">
          <div class="form-group">
            <label class="form-label">Student ID</label>
            <input 
              v-model="form.studentId" 
              type="text" 
              class="form-input" 
              placeholder="Enter student ID"
              required
            />
          </div>

          <div class="upload-sections">
            <div class="upload-section">
              <h3>
                <span class="upload-icon">🎤</span>
                Speech Analysis
              </h3>
              <p class="upload-hint">Upload audio recording (WAV, MP3)</p>
              <input 
                type="file" 
                accept="audio/*"
                class="file-input"
                @change="handleAudioUpload"
              />
            </div>

            <div class="upload-section">
              <h3>
                <span class="upload-icon">✍️</span>
                Handwriting Analysis
              </h3>
              <p class="upload-hint">Upload handwriting sample image</p>
              <input 
                type="file" 
                accept="image/*"
                class="file-input"
                @change="handleImageUpload"
              />
            </div>

            <div class="upload-section">
              <h3>
                <span class="upload-icon">📝</span>
                Text Analysis
              </h3>
              <p class="upload-hint">Enter text written by student</p>
              <textarea 
                v-model="form.text"
                class="form-input text-input"
                placeholder="Enter the student's written text here..."
                rows="4"
              ></textarea>
            </div>
          </div>

          <button 
            type="submit" 
            class="btn btn-primary submit-btn"
            :disabled="isLoading"
          >
            <span v-if="isLoading" class="loading-spinner"></span>
            <span v-else>Analyze</span>
          </button>
        </form>
      </div>

      <div class="results-card card" v-if="results">
        <h2>Analysis Results</h2>
        
        <div class="risk-level">
          <span class="risk-label">Overall Risk:</span>
          <span :class="['badge', `badge-${results.overall_risk}`]">
            {{ results.overall_risk.toUpperCase() }}
          </span>
          <span class="risk-score">({{ (results.overall_score * 100).toFixed(1) }}%)</span>
        </div>

        <div class="score-breakdown">
          <div class="score-item">
            <div class="score-header">
              <span>🎤 Speech</span>
              <span class="score-value">{{ (results.speech_score * 100).toFixed(1) }}%</span>
            </div>
            <div class="score-bar">
              <div class="score-fill" :style="{ width: `${results.speech_score * 100}%` }"></div>
            </div>
          </div>

          <div class="score-item">
            <div class="score-header">
              <span>✍️ Handwriting</span>
              <span class="score-value">{{ (results.handwriting_score * 100).toFixed(1) }}%</span>
            </div>
            <div class="score-bar">
              <div class="score-fill" :style="{ width: `${results.handwriting_score * 100}%` }"></div>
            </div>
          </div>

          <div class="score-item">
            <div class="score-header">
              <span>📝 Text</span>
              <span class="score-value">{{ (results.text_score * 100).toFixed(1) }}%</span>
            </div>
            <div class="score-bar">
              <div class="score-fill" :style="{ width: `${results.text_score * 100}%` }"></div>
            </div>
          </div>
        </div>

        <div class="explanation" v-if="results.explanation">
          <h3>Explanation</h3>
          <p>{{ results.explanation }}</p>
        </div>

        <div class="features" v-if="results.top_features && results.top_features.length">
          <h3>Top Features</h3>
          <div v-for="modality in results.top_features" :key="modality.modality" class="feature-group">
            <h4>{{ modality.modality }}</h4>
            <ul>
              <li v-for="(value, key) in modality.features" :key="key">
                {{ key }}: {{ typeof value === 'number' ? value.toFixed(3) : value }}
              </li>
            </ul>
          </div>
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
.hero {
  text-align: center;
  margin-bottom: 2rem;
}

.hero h1 {
  font-size: 2rem;
  color: var(--gray-800);
  margin-bottom: 0.5rem;
}

.hero p {
  color: var(--gray-500);
}

.content-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
}

@media (max-width: 1024px) {
  .content-grid {
    grid-template-columns: 1fr;
  }
}

.analysis-card h2,
.results-card h2 {
  margin-bottom: 1.5rem;
  color: var(--gray-800);
}

.upload-sections {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  margin-bottom: 1.5rem;
}

.upload-section {
  padding: 1rem;
  background: var(--gray-50);
  border-radius: 8px;
}

.upload-section h3 {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 1rem;
  margin-bottom: 0.5rem;
}

.upload-icon {
  font-size: 1.25rem;
}

.upload-hint {
  font-size: 0.875rem;
  color: var(--gray-500);
  margin-bottom: 0.75rem;
}

.text-input {
  resize: vertical;
  min-height: 100px;
}

.submit-btn {
  width: 100%;
  padding: 1rem;
  font-size: 1.125rem;
}

.risk-level {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  background: var(--gray-50);
  border-radius: 8px;
  margin-bottom: 1.5rem;
}

.risk-label {
  font-weight: 600;
}

.risk-score {
  color: var(--gray-500);
}

.score-breakdown {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.score-item {
  padding: 0.75rem;
  background: var(--gray-50);
  border-radius: 8px;
}

.score-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.5rem;
  font-weight: 500;
}

.score-value {
  color: var(--primary);
}

.score-bar {
  height: 8px;
  background: var(--gray-200);
  border-radius: 4px;
  overflow: hidden;
}

.score-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--primary) 0%, var(--primary-dark) 100%);
  border-radius: 4px;
  transition: width 0.5s ease;
}

.explanation {
  padding: 1rem;
  background: #eff6ff;
  border-radius: 8px;
  border-left: 4px solid var(--primary);
  margin-bottom: 1.5rem;
}

.explanation h3 {
  margin-bottom: 0.5rem;
  font-size: 1rem;
}

.explanation p {
  color: var(--gray-700);
  line-height: 1.7;
}

.features {
  padding: 1rem;
  background: var(--gray-50);
  border-radius: 8px;
}

.features h3 {
  margin-bottom: 1rem;
}

.feature-group {
  margin-bottom: 1rem;
}

.feature-group h4 {
  text-transform: capitalize;
  margin-bottom: 0.5rem;
  color: var(--primary);
}

.feature-group ul {
  list-style: none;
  font-size: 0.875rem;
}

.feature-group li {
  padding: 0.25rem 0;
  color: var(--gray-600);
}
</style>
