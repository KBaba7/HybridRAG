let documents = [];
let activeDoc = null;
let chatHistory = [];
let mode = 'chat';
let msgCounter = 0;

document.addEventListener('DOMContentLoaded', () => {
  loadExistingDocuments();
});

async function loadExistingDocuments() {
  try {
    const resp = await fetch('http://localhost:8000/api/documents');
    if (!resp.ok) return;
    const data = await resp.json();
    if (!data.documents || data.documents.length === 0) return;

    documents = data.documents;
    activeDoc = documents[0];

    renderDocList();
    renderStats();
    enableChat();
    document.getElementById('empty-state').style.display = 'none';
    showNotif(`Restored ${documents.length} document${documents.length > 1 ? 's' : ''} from previous session`, 'success');
  } catch (e) {
    // backend may not be up yet — fail silently
  }
}


const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');

uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => { e.preventDefault(); uploadZone.classList.remove('drag-over'); handleFiles(e.dataTransfer.files); });
fileInput.addEventListener('change', e => handleFiles(e.target.files));

async function handleFiles(files) {
  for (const file of files) {
    if (file.type !== 'application/pdf') { showNotif('Only PDF files supported', 'error'); continue; }
    await ingestPDF(file);
  }
}

async function ingestPDF(file) {
  showProgress(30, `Uploading ${file.name} to spatial parser...`);
  
  const docId = Date.now().toString();
  const companyName = file.name.replace('.pdf', '');

  const formData = new FormData();
  formData.append('file', file);
  formData.append('doc_id', docId);
  formData.append('company_name', companyName);

  try {
    const response = await fetch('http://localhost:8000/api/ingest', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error('Failed to ingest document');
    
    const data = await response.json();
    
    const doc = {
      id: docId,
      name: companyName,
      pages: data.pages_processed,
      tableCount: data.tables_detected
    };

    documents.push(doc);
    activeDoc = doc;
    
    showProgress(100, 'Ready!');
    setTimeout(() => { document.getElementById('progress-bar').style.display = 'none'; document.getElementById('progress-text').textContent = ''; }, 1200);

    renderDocList();
    renderStats();
    enableChat();
    showNotif(`✓ Parsed ${doc.name} — ${doc.pages} pages, ${doc.tableCount} tables identified`, 'success');
    document.getElementById('empty-state').style.display = 'none';

  } catch (error) {
    showProgress(0, '');
    showNotif(`Error: ${error.message}`, 'error');
  }
}

function showProgress(pct, msg) {
  const bar = document.getElementById('progress-bar');
  bar.style.display = 'block';
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('progress-text').textContent = msg;
}

function renderDocList() {
  document.getElementById('docs-section').style.display = 'block';
  const list = document.getElementById('doc-list');
  list.innerHTML = documents.map(d => `
    <div class="doc-item ${d.id === (activeDoc && activeDoc.id) ? 'active' : ''}" onclick="selectDoc('${d.id}')">
      <div class="doc-icon">10-K</div>
      <div class="doc-meta">
        <div class="doc-name">${d.name}</div>
        <div class="doc-pages">${d.pages} pages</div>
      </div>
      <div class="doc-status ready"></div>
    </div>`).join('');
}

function renderStats() {
  const totalPages = documents.reduce((sum, d) => sum + d.pages, 0);
  const totalTables = documents.reduce((sum, d) => sum + d.tableCount, 0);
  document.getElementById('stats-section').style.display = 'block';
  document.getElementById('stats-grid').innerHTML = `
    <div class="stat-card"><div class="stat-label">Pages</div><div class="stat-val">${totalPages}</div></div>
    <div class="stat-card"><div class="stat-label">Tables</div><div class="stat-val green">${totalTables}</div></div>
  `;
  document.getElementById('token-info').textContent = `${documents.length} doc${documents.length !== 1 ? 's' : ''} · RAG Enabled`;
}

function selectDoc(id) {
  activeDoc = documents.find(d => d.id === id);
  renderDocList();
}

function enableChat() {
  document.getElementById('chat-input').disabled = false;
  document.getElementById('send-btn').disabled = false;
  document.getElementById('chat-input').placeholder = 'Ask about financials...';
}

function setMode(m) {
  mode = m;
  ['chat', 'extract', 'compare'].forEach(x => {
    document.getElementById(`btn-${x}`).classList.toggle('active', x === m);
  });
  
  if (m === 'compare' && documents.length < 2) {
    showNotif('Upload at least 2 documents to compare', 'error');
    setMode('chat');
  }
}

function sendQuick(text) {
  if (documents.length === 0) { showNotif('Please upload a document first', 'error'); return; }
  document.getElementById('chat-input').value = text;
  sendMessage();
}

const chatInput = document.getElementById('chat-input');
chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text || documents.length === 0) return;

  chatInput.value = '';
  addMsg('user', text);
  const thinkingId = addThinking();

  try {
    // always search all docs; compare mode is already all docs too
    const docIdsToSearch = documents.map(d => d.id);
    
    const resp = await fetch('http://localhost:8000/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: chatHistory,
        user_message: text,
        active_doc_ids: docIdsToSearch
      })
    });

    if (!resp.ok) {
      let message = 'API Error';
      try {
        const errorData = await resp.json();
        message = errorData.detail || message;
      } catch (e) {}
      throw new Error(message);
    }
    const data = await resp.json();
    
    removeThinking(thinkingId);
    renderAssistantResponse(data.response, data.tools_used || []);
    
    chatHistory.push({ role: 'user', content: text });
    chatHistory.push({ role: 'assistant', content: data.response });
    if (chatHistory.length > 10) chatHistory = chatHistory.slice(-10);

  } catch (err) {
    removeThinking(thinkingId);
    addMsg('assistant', `Error processing request: ${escHtml(err.message)}`);
  }
}

function renderAssistantResponse(text, toolsUsed = []) {
  const chartMatch = text.match(/```chart\s*([\s\S]*?)```/);
  if (chartMatch) {
    try {
      const chartDef = JSON.parse(chartMatch[1].trim());
      const beforeChart = text.slice(0, chartMatch.index);
      const afterChart = text.slice(chartMatch.index + chartMatch[0].length);
      const msgId = addMsg('assistant', formatMsgText(beforeChart + afterChart));
      renderChart(chartDef, msgId);
      renderToolTrace(toolsUsed);
      return;
    } catch (e) {}
  }
  addMsg('assistant', formatMsgText(text));
  renderToolTrace(toolsUsed);
}

const TOOL_META = {
  select_documents: { label: 'select_documents' },
  retrieve_chunks:  { label: 'retrieve_chunks' },
  generate_chart:   { label: 'generate_chart' },
};

function renderToolTrace(toolsUsed) {
  if (!toolsUsed || toolsUsed.length === 0) return;
  const container = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg assistant tool-trace-row';
  const pills = toolsUsed.map(name => {
    const meta = TOOL_META[name] || { label: name };
    return `<span class="tool-pill">${meta.label}</span>`;
  }).join('');
  div.innerHTML = `<div class="msg-avatar" style="visibility:hidden">A</div><div class="tool-trace">${pills}</div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function parseMarkdownTable(text) {
  // Match a markdown table block: header row | separator row | data rows
  return text.replace(/(\|.+\|\n\|[-| :]+\|\n(?:\|.+\|\n?)+)/g, (match) => {
    const lines = match.trim().split('\n').filter(l => l.trim());
    if (lines.length < 2) return match;

    const headers = lines[0].split('|').slice(1, -1).map(h => h.trim());
    // skip separator line (index 1)
    const rows = lines.slice(2).map(l => l.split('|').slice(1, -1).map(c => c.trim()));

    const thead = headers.map(h => `<th>${escHtml(h)}</th>`).join('');
    const tbody = rows.map(row =>
      '<tr>' + row.map(cell => {
        const isNum = /^-?[\d,.$%]+$/.test(cell.replace(/\s/g, ''));
        const cls = isNum ? ' class="num"' : '';
        return `<td${cls}>${escHtml(cell)}</td>`;
      }).join('') + '</tr>'
    ).join('');

    return `<table class="msg-table"><thead><tr>${thead}</tr></thead><tbody>${tbody}</tbody></table>`;
  });
}

function formatMsgText(text) {
  // Convert markdown tables before other formatting
  let formatted = parseMarkdownTable(text);

  formatted = formatted
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>');

  if (/<table[\s>]/i.test(formatted)) {
    return formatted;
  }

  return formatted
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>')
    .replace(/^/, '<p>').replace(/$/, '</p>');
}

function addMsg(role, html) {
  const id = 'msg-' + (++msgCounter);
  const container = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.id = id;
  div.innerHTML = `<div class="msg-avatar">${role === 'user' ? 'U' : 'A'}</div><div class="msg-bubble">${role === 'user' ? escHtml(html) : html}</div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return id;
}

function addThinking() {
  const id = 'thinking-' + Date.now();
  const container = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.id = id;
  div.innerHTML = `<div class="msg-avatar">A</div><div class="msg-bubble"><div class="thinking"><div class="dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>Analyzing document chunks...</div></div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return id;
}

function removeThinking(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function renderChart(def, afterMsgId) {
  const container = document.getElementById('messages');
  const chartDiv = document.createElement('div');
  chartDiv.className = 'msg assistant';
  chartDiv.innerHTML = `<div class="msg-avatar">A</div><div class="msg-bubble"><div class="chart-card"><div class="chart-title">${def.title || 'Chart'}</div><div style="position:relative; width:100%; height:260px"><canvas></canvas></div></div></div>`;
  container.appendChild(chartDiv);
  container.scrollTop = container.scrollHeight;

  setTimeout(() => {
    const canvas = chartDiv.querySelector('canvas');
    new Chart(canvas, {
      type: def.type || 'bar',
      data: { labels: def.labels, datasets: def.datasets },
      options: { responsive: true, maintainAspectRatio: false }
    });
  }, 100);
}

function escHtml(str) { return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function showNotif(msg, type = '') {
  const n = document.getElementById('notif');
  n.textContent = msg;
  n.className = 'notif show ' + type;
  setTimeout(() => n.className = 'notif', 3500);
}
