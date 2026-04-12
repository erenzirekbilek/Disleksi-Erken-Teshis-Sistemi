import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json'
  }
})

export const analysisService = {
  async analyze(studentId, audioFile = null, imageFile = null, text = null) {
    const formData = new FormData()
    formData.append('student_id', studentId)
    
    if (audioFile) {
      formData.append('audio', audioFile)
    }
    if (imageFile) {
      formData.append('image', imageFile)
    }
    if (text) {
      formData.append('text', text)
    }

    const response = await api.post('/analyze', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    return response.data
  },

  async analyzeAudio(studentId, audioFile) {
    const formData = new FormData()
    formData.append('student_id', studentId)
    formData.append('audio', audioFile)

    const response = await api.post('/analyze/audio', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    return response.data
  },

  async analyzeHandwriting(studentId, imageFile) {
    const formData = new FormData()
    formData.append('student_id', studentId)
    formData.append('image', imageFile)

    const response = await api.post('/analyze/handwriting', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    return response.data
  },

  async analyzeText(studentId, text) {
    const formData = new FormData()
    formData.append('student_id', studentId)
    formData.append('text', text)

    const response = await api.post('/analyze/text', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    return response.data
  },

  async explain(speechScore, handwritingScore, textScore) {
    const response = await api.get('/explain', {
      params: {
        speech_score: speechScore,
        handwriting_score: handwritingScore,
        text_score: textScore
      }
    })
    return response.data
  },

  async getHealth() {
    const response = await api.get('/health')
    return response.data
  }
}

export default api
