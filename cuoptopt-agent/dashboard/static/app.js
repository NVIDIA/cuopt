// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';

// ---------------------------------------------------------------------------
// Constants & state
// ---------------------------------------------------------------------------

const SESSION_KEY = 'cuopt_session_id';
const METRICS_KEY = 'cuopt_last_metrics';

let socket = null;
let isRunning = false;
let sessionId = null;

// Resolve session ID: prefer cookie (set by server), fall back to localStorage
function getSessionId() {
  const cookie = document.cookie.split(';')
    .map(c => c.trim())
    .find(c => c.startsWith('sid='));
  if (cookie) {
    const id = cookie.split('=')[1];
    localStorage.setItem(SESSION_KEY, id);
    return id;
  }
  return localStorage.getItem(SESSION_KEY) || null;
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function scrollBottom() {
  const el = document.getElementById('messages');
  el.scrollTop = el.scrollHeight;
}

function setStatus(text, cls) {
  const badge = document.getElementById('status-badge');
  badge.textContent = text;
  badge.className = `status-badge ${cls}`;
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

function openSocket() {
  if (socket && socket.readyState <= WebSocket.OPEN) return;

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const sid = sessionId || 'anon';
  socket = new WebSocket(`${proto}://${location.host}/ws/${sid}`);

  socket.addEventListener('open', () => {
    setStatus('Connected', 'connected');
    // Keep-alive ping every 20 s
    socket._ping = setInterval(() => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'ping' }));
      }
    }, 20_000);
  });

  socket.addEventListener('close', () => {
    clearInterval(socket._ping);
    setStatus('Disconnected', 'idle');
    setRunning(false);
    // Auto-reconnect after 3 s
    setTimeout(openSocket, 3_000);
  });

  socket.addEventListener('error', () => {
    setStatus('Error', 'error');
  });

  socket.addEventListener('message', evt => {
    try {
      const msg = JSON.parse(evt.data);
      handleFrame(msg);
    } catch {/* ignore malformed frames */}
  });
}

function sendFrame(payload) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(payload));
  }
}

// ---------------------------------------------------------------------------
// Frame handlers
// ---------------------------------------------------------------------------

function handleFrame(msg) {
  switch (msg.type) {
    case 'section':
      appendSection(msg.title);
      break;
    case 'log':
      appendLog(msg.text);
      break;
    case 'approval_request':
      appendApprovalCard(msg.prompt);
      break;
    case 'metrics':
      renderMetrics(msg.data);
      break;
    case 'pr_url':
      appendAgentMessage(`Pull request created: <a href="${escapeHtml(msg.url)}" target="_blank" rel="noopener">${escapeHtml(msg.url)}</a>`);
      break;
    case 'done':
      setRunning(false);
      setStatus(msg.success ? 'Done' : 'Failed', msg.success ? 'done' : 'error');
      if (msg.success) {
        appendAgentMessage('Run completed successfully.');
      } else {
        appendAgentMessage('Run ended — no changes committed.');
      }
      break;
    case 'error':
      appendAgentMessage(`<span class="error-text">Error: ${escapeHtml(msg.text)}</span>`);
      setRunning(false);
      setStatus('Error', 'error');
      break;
    case 'pong':
      break;
    case '_eof':
      break;
  }
  scrollBottom();
}

// ---------------------------------------------------------------------------
// Message rendering helpers
// ---------------------------------------------------------------------------

function cloneTemplate(id) {
  return document.getElementById(id).content.cloneNode(true);
}

function appendSection(title) {
  const frag = cloneTemplate('tmpl-section');
  frag.querySelector('.section-label').textContent = title;
  document.getElementById('messages').appendChild(frag);
}

function appendLog(text) {
  // Collapse many consecutive log lines into a collapsible block
  const feed = document.getElementById('messages');
  let logBlock = feed.lastElementChild;
  if (!logBlock || !logBlock.classList.contains('log-block')) {
    logBlock = document.createElement('details');
    logBlock.className = 'log-block';
    const summary = document.createElement('summary');
    summary.textContent = 'Agent output';
    logBlock.appendChild(summary);
    feed.appendChild(logBlock);
  }
  const frag = cloneTemplate('tmpl-log');
  frag.querySelector('code').textContent = text;
  logBlock.appendChild(frag);
}

function appendAgentMessage(html) {
  const frag = cloneTemplate('tmpl-msg');
  const el = frag.querySelector('.msg');
  el.dataset.role = 'agent';
  el.querySelector('.msg-role').textContent = 'Agent';
  el.querySelector('.msg-body').innerHTML = html;
  document.getElementById('messages').appendChild(frag);
}

function appendUserMessage(text) {
  const frag = cloneTemplate('tmpl-msg');
  const el = frag.querySelector('.msg');
  el.dataset.role = 'user';
  el.querySelector('.msg-role').textContent = 'You';
  el.querySelector('.msg-body').textContent = text;
  document.getElementById('messages').appendChild(frag);
}

function appendApprovalCard(prompt) {
  const frag = cloneTemplate('tmpl-approval');
  const card = frag.querySelector('.approval-card');
  card.querySelector('.approval-prompt').textContent = prompt;

  const acceptBtn = card.querySelector('.btn-accept');
  const denyBtn = card.querySelector('.btn-deny');

  function respond(value) {
    sendFrame({ type: 'approve', value });
    acceptBtn.disabled = true;
    denyBtn.disabled = true;
    card.classList.add(value ? 'approved' : 'denied');
    card.querySelector('.approval-actions').innerHTML =
      `<span class="approval-result">${value ? '✓ Accepted' : '✗ Denied'}</span>`;
  }

  acceptBtn.addEventListener('click', () => respond(true));
  denyBtn.addEventListener('click', () => respond(false));

  document.getElementById('messages').appendChild(frag);
  scrollBottom();
}

// ---------------------------------------------------------------------------
// Query submission
// ---------------------------------------------------------------------------

function setRunning(val) {
  isRunning = val;
  const btn = document.getElementById('send-btn');
  const input = document.getElementById('query-input');
  btn.disabled = val;
  input.disabled = val;
  if (val) setStatus('Running…', 'running');
}

document.getElementById('query-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  if (isRunning) return;

  const input = document.getElementById('query-input');
  const query = input.value.trim();
  if (!query) return;

  const model = document.getElementById('model-select').value;

  appendUserMessage(query);
  input.value = '';
  scrollBottom();

  // Persist user message to history via the server-side session
  fetch('/api/history', { method: 'GET' }); // just touches the endpoint to refresh cookie

  setRunning(true);
  openSocket(); // ensure connected
  sendFrame({ type: 'query', query, model });
});

// Ctrl+Enter submits
document.getElementById('query-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    document.getElementById('query-form').dispatchEvent(new Event('submit'));
  }
});

// ---------------------------------------------------------------------------
// Branch sidebar
// ---------------------------------------------------------------------------

async function loadBranches() {
  const list = document.getElementById('branch-list');
  list.innerHTML = '<div class="branch-skeleton">Loading…</div>';

  try {
    const resp = await fetch('/api/branches');
    if (!resp.ok) throw new Error(await resp.text());
    const branches = await resp.json();

    if (branches.error) {
      list.innerHTML = `<div class="branch-error">${escapeHtml(branches.error)}</div>`;
      return;
    }

    list.innerHTML = '';
    for (const b of branches) {
      list.appendChild(buildBranchCard(b));
    }
  } catch (err) {
    list.innerHTML = `<div class="branch-error">Failed to load branches: ${escapeHtml(String(err))}</div>`;
  }
}

function buildBranchCard(b) {
  const frag = cloneTemplate('tmpl-branch');
  const card = frag.querySelector('.branch-card');
  card.dataset.name = b.name;

  card.querySelector('.branch-name').textContent = b.name;
  card.querySelector('.branch-date').textContent = b.date ? b.date.slice(0, 10) : '';

  const prLink = card.querySelector('.branch-pr-link');
  if (b.pr_url) {
    prLink.href = b.pr_url;
    prLink.hidden = false;
  }

  const stat = card.querySelector('.branch-delta');
  const instances = b.metrics && b.metrics.instances ? b.metrics.instances : [];

  if (instances.length) {
    // Compute average solve time from this branch's own recorded times
    const times = instances.map(i => i.candidate_s).filter(t => t != null);
    const avg = times.length ? times.reduce((a, v) => a + v, 0) / times.length : null;
    stat.textContent = avg != null ? `avg ${avg.toFixed(2)} s` : 'no data';
    stat.className = 'branch-delta neutral';

    // Per-instance breakdown table (two columns: Instance | Time)
    const instancesDiv = card.querySelector('.branch-instances');
    instancesDiv.classList.remove('hidden');

    const table = document.createElement('table');
    table.className = 'instance-table';
    table.innerHTML = '<thead><tr><th>Instance</th><th>Time (s)</th></tr></thead>';
    const tbody = document.createElement('tbody');

    for (const inst of instances) {
      const row = document.createElement('tr');
      row.innerHTML = `
        <td title="${escapeHtml(inst.name)}">${escapeHtml(inst.name.replace('.mps', ''))}</td>
        <td>${inst.candidate_s != null ? inst.candidate_s.toFixed(2) : '—'}</td>`;
      tbody.appendChild(row);
    }
    table.appendChild(tbody);
    instancesDiv.appendChild(table);

    // Tap card header: toggle instance table + open plot
    const header = card.querySelector('.branch-card-header');
    header.style.cursor = 'pointer';
    header.addEventListener('click', () => {
      instancesDiv.classList.toggle('hidden');
      renderMetrics(b.metrics, b.name);
    });
  } else {
    stat.textContent = 'no data';
    stat.className = 'branch-delta neutral';
  }

  return frag;
}

document.getElementById('refresh-branches').addEventListener('click', loadBranches);

// Auto-refresh branches every 60 s
setInterval(loadBranches, 60_000);

// ---------------------------------------------------------------------------
// Performance plot (Plotly)
// ---------------------------------------------------------------------------

function renderMetrics(data, label) {
  const panel = document.getElementById('plot-panel');
  panel.classList.add('visible');

  if (!data || !data.instances || !data.instances.length) return;

  // Update panel title with branch name
  const titleEl = document.getElementById('plot-title');
  if (titleEl) titleEl.textContent = label ? `${label} — solve times` : 'Branch performance';

  // Persist to sessionStorage so reload restores the chart
  try {
    sessionStorage.setItem(METRICS_KEY, JSON.stringify({ data, label }));
  } catch { /* quota exceeded — ignore */ }

  const instances = data.instances.filter(i => i.candidate_s != null);
  // Horizontal bar: instances on Y-axis, time on X-axis
  const names = instances.map(i => i.name.replace('.mps', ''));
  const times = instances.map(i => i.candidate_s);

  const trace = {
    y: names,
    x: times,
    type: 'bar',
    orientation: 'h',
    name: label || 'branch',
    marker: { color: 'rgba(99,179,237,0.85)' },
    text: times.map(t => `${t.toFixed(2)} s`),
    textposition: 'outside',
    cliponaxis: false,
  };

  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#e2e8f0', family: 'system-ui' },
    xaxis: { title: 'Solve time (s)', gridcolor: '#334155', zeroline: false },
    yaxis: { gridcolor: '#334155', automargin: true },
    margin: { t: 20, r: 60, b: 50, l: 20 },
    showlegend: false,
  };

  Plotly.newPlot('plot', [trace], layout, {
    responsive: true,
    displayModeBar: false,
  });
}

document.getElementById('close-plot').addEventListener('click', () => {
  document.getElementById('plot-panel').classList.remove('visible');
});

// Restore last chart from sessionStorage on page load
function restoreMetrics() {
  try {
    const raw = sessionStorage.getItem(METRICS_KEY);
    if (raw) {
      const { data, label } = JSON.parse(raw);
      renderMetrics(data, label);
    }
  } catch { /* ignore */ }
}

// ---------------------------------------------------------------------------
// Chat history replay
// ---------------------------------------------------------------------------

async function loadHistory() {
  try {
    const resp = await fetch('/api/history');
    if (!resp.ok) return;
    const payload = await resp.json();
    sessionId = payload.session_id;

    const feed = document.getElementById('messages');
    if (!payload.messages || !payload.messages.length) return;

    for (const msg of payload.messages) {
      if (msg.role === 'user') {
        appendUserMessage(msg.content);
      } else {
        appendAgentMessage(escapeHtml(msg.content));
      }
    }
    scrollBottom();
  } catch { /* silently ignore */ }
}

// ---------------------------------------------------------------------------
// Mobile sidebar toggle
// ---------------------------------------------------------------------------

document.getElementById('hamburger').addEventListener('click', () => {
  document.getElementById('sidebar').classList.toggle('open');
  document.getElementById('overlay').classList.toggle('visible');
});

document.getElementById('overlay').addEventListener('click', () => {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('overlay').classList.remove('visible');
});

// ---------------------------------------------------------------------------
// Boot sequence
// ---------------------------------------------------------------------------

(async function init() {
  sessionId = getSessionId();
  await loadHistory();
  loadBranches();
  restoreMetrics();
  openSocket();
})();
