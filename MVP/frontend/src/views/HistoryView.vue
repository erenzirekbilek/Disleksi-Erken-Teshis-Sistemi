<template>
  <div class="history">
    <div class="hero">
      <h1>Assessment History</h1>
      <p>View past assessments and track progress over time</p>
    </div>

    <div class="card">
      <div class="empty-state">
        <span class="empty-icon">📋</span>
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

.empty-state {
  text-align: center;
  padding: 3rem;
}

.empty-icon {
  font-size: 4rem;
  display: block;
  margin-bottom: 1rem;
}

.empty-state h3 {
  margin-bottom: 0.5rem;
  color: var(--gray-700);
}

.empty-state p {
  color: var(--gray-500);
  margin-bottom: 1.5rem;
}
</style>
