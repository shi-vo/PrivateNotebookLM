// Plain vanilla JS — no frameworks, no build step. Every call below talks
// to the FastAPI backend defined in main.py, same-origin.

const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatusList = document.getElementById("uploadStatusList");
const docSelect = document.getElementById("docSelect");
const docList = document.getElementById("docList");
const healthBanner = document.getElementById("healthBanner");
const chatWindow = document.getElementById("chatWindow");
const chatEmptyState = document.getElementById("chatEmptyState");
const thinkingIndicator = document.getElementById("thinkingIndicator");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");

document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  loadDocuments();
});

// --------------------------------------------------------------------
// Health check — shown as a banner if Ollama or a required model isn't
// ready, with the exact commands the user needs to run.
// --------------------------------------------------------------------
async function checkHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();

    if (data.ok) {
      healthBanner.classList.add("hidden");
      healthBanner.innerHTML = "";
      return;
    }

    let message;
    if (!data.ollama_reachable) {
      message = "Ollama isn't reachable. Start it, then run the model pulls below:";
    } else {
      message = "Some required Ollama models aren't pulled yet:";
    }

    const commandsHtml = (data.missing_commands || [])
      .map((cmd) => `<code>${escapeHtml(cmd)}</code>`)
      .join(" ");

    healthBanner.innerHTML = `<strong>Setup needed.</strong> ${message}<br>${commandsHtml}`;
    healthBanner.classList.remove("hidden");
  } catch (err) {
    healthBanner.innerHTML =
      "<strong>Could not reach the backend.</strong> Is the server running? (`uvicorn main:app --reload`)";
    healthBanner.classList.remove("hidden");
  }
}

// --------------------------------------------------------------------
// Documents: populate dropdown + list, used on load and after every
// upload/delete so the UI never needs a full page reload.
// --------------------------------------------------------------------
async function loadDocuments() {
  let docs = [];
  try {
    const res = await fetch("/documents");
    docs = await res.json();
  } catch (err) {
    console.error("Failed to load documents", err);
    return;
  }

  const previousSelection = docSelect.value;

  docSelect.innerHTML = "";
  const allOption = document.createElement("option");
  allOption.value = "all";
  allOption.textContent = "All Documents";
  docSelect.appendChild(allOption);

  for (const doc of docs) {
    const opt = document.createElement("option");
    opt.value = doc.doc_id;
    opt.textContent = doc.filename;
    docSelect.appendChild(opt);
  }

  // Preserve the user's current selection if that document still exists.
  const stillExists = Array.from(docSelect.options).some((o) => o.value === previousSelection);
  docSelect.value = stillExists ? previousSelection : "all";

  renderDocList(docs);
}

function renderDocList(docs) {
  docList.innerHTML = "";
  if (docs.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No documents uploaded yet.";
    docList.appendChild(li);
    return;
  }

  for (const doc of docs) {
    const li = document.createElement("li");

    const meta = document.createElement("span");
    meta.className = "doc-meta";
    meta.textContent = `${doc.filename} — ${doc.chunk_count} chunk(s)`;

    const delBtn = document.createElement("button");
    delBtn.className = "delete-btn";
    delBtn.type = "button";
    delBtn.textContent = "Remove";
    delBtn.addEventListener("click", () => deleteDocument(doc.doc_id, doc.filename));

    li.appendChild(meta);
    li.appendChild(delBtn);
    docList.appendChild(li);
  }
}

async function deleteDocument(docId, filename) {
  if (!confirm(`Remove "${filename}" and all of its indexed chunks?`)) return;
  try {
    const res = await fetch(`/documents/${docId}`, { method: "DELETE" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(`Failed to remove document: ${err.detail || res.statusText}`);
      return;
    }
    await loadDocuments();
  } catch (err) {
    alert(`Failed to remove document: ${err}`);
  }
}

// --------------------------------------------------------------------
// Upload
// --------------------------------------------------------------------
uploadBtn.addEventListener("click", async () => {
  const files = Array.from(fileInput.files || []);
  if (files.length === 0) {
    alert("Choose at least one .pdf, .docx, or .txt file first.");
    return;
  }

  uploadBtn.disabled = true;
  for (const file of files) {
    await uploadOneFile(file);
  }
  uploadBtn.disabled = false;
  fileInput.value = "";

  // Refresh dropdown + doc list to include the newly uploaded doc(s)
  // without a full page reload.
  await loadDocuments();
});

async function uploadOneFile(file) {
  const statusLi = document.createElement("li");
  statusLi.textContent = `Uploading "${file.name}"…`;
  uploadStatusList.prepend(statusLi);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      statusLi.textContent = `"${file.name}" failed: ${data.detail || res.statusText}`;
      statusLi.className = "error";
      return;
    }

    statusLi.textContent = `"${data.filename}" indexed — ${data.chunk_count} chunk(s).`;
    statusLi.className = "ok";
  } catch (err) {
    statusLi.textContent = `"${file.name}" failed: ${err}`;
    statusLi.className = "error";
  }
}

// --------------------------------------------------------------------
// Chat
// --------------------------------------------------------------------
chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  chatEmptyState.remove();
  addBubble("user", question);
  questionInput.value = "";
  questionInput.disabled = true;
  thinkingIndicator.classList.remove("hidden");
  scrollChatToBottom();

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, doc_id: docSelect.value }),
    });
    const data = await res.json();

    if (!res.ok) {
      addBubble("error", data.detail || "Something went wrong.");
    } else {
      addBubble("assistant", data.answer);
      addCitations(data.citations || []);
    }
  } catch (err) {
    addBubble("error", `Could not reach the backend: ${err}`);
  } finally {
    thinkingIndicator.classList.add("hidden");
    questionInput.disabled = false;
    questionInput.focus();
    scrollChatToBottom();
  }
});

function addBubble(role, text) {
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  div.textContent = text;
  chatWindow.appendChild(div);
  scrollChatToBottom();
}

function addCitations(citations) {
  if (!citations || citations.length === 0) return;

  const container = document.createElement("div");
  container.className = "citations";

  for (const c of citations) {
    const details = document.createElement("details");
    details.className = "citation";

    const summary = document.createElement("summary");
    summary.textContent = `📄 ${c.doc_name} — ${c.page_or_section}`;

    const excerpt = document.createElement("div");
    excerpt.className = "excerpt";
    excerpt.textContent = c.excerpt;

    details.appendChild(summary);
    details.appendChild(excerpt);
    container.appendChild(details);
  }

  chatWindow.appendChild(container);
  scrollChatToBottom();
}

function scrollChatToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
