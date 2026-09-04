// API Client for ClauseGuard Backend
// Configured to point to custom backend URL (Vercel deployment) or local proxy

const BASE_URL = import.meta.env.VITE_API_URL 
  ? import.meta.env.VITE_API_URL.replace(/\/$/, '') 
  : '';

export async function analyzeContract({ file, text }) {
  const formData = new FormData();
  if (file) {
    formData.append('file', file);
  }
  if (text) {
    formData.append('text', text);
  }

  const endpoint = `${BASE_URL}/api/analyze`;
  
  const response = await fetch(endpoint, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Analysis failed with status ${response.status}`);
  }

  return response.json();
}

export async function chatWithContract({ question, clauses }) {
  const endpoint = `${BASE_URL}/api/chat`;

  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question, clauses }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Chat request failed with status ${response.status}`);
  }

  return response.json();
}

export async function checkBackendHealth() {
  const endpoint = `${BASE_URL}/health`;
  try {
    const res = await fetch(endpoint);
    return res.ok;
  } catch {
    return false;
  }
}
