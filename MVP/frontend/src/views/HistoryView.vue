<template>
  <div class="history">
    <header class="page-header">
      <h1>Assessment History</h1>
      <p>View past assessments and track progress over time</p>
    </header>

    <div class="card">
      <div class="empty-state">
        <div class="empty-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M9 5H7a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2"/>
            <rect x="9" y="3" width="6" height="4" rx="1"/>
            <path d="M9 12h6M9 16h6"/>
          </svg>
        </div>
        <h3>No History Available</h3>
        <p>Assessment history will appear here after the first analysis.</p>
        <router-link to="/" class="btn btn-primary">
          Start New Assessment
        </router-link>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'HistoryView'
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

.empty-state {
  text-align: center;
  padding: 4rem 2rem;
}

.empty-icon {
  width: 80px;
  height: 80px;
  background: var(--gray-100);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 1.5rem;
}

.empty-icon svg {
  width: 40px;
  height: 40px;
  color: var(--gray-400);
}

.empty-state h3 {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--gray-800);
  margin-bottom: 0.5rem;
}

.empty-state p {
  color: var(--gray-500);
  margin-bottom: 2rem;
  max-width: 300px;
  margin-left: auto;
  margin-right: auto;
}
</style>