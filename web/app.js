// Barbershop Assistant — minimal chat client.
// Keeps the conversation history in memory and sends it on every turn.
// Stateless server, stateful client.

const chatEl = document.getElementById("chat");
const formEl = document.getElementById("form");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const statusEl = document.getElementById("status");
const modelEl = document.getElementById("model");

const history = []; // [{role, content}, ...]

function renderMessage(role, content, meta) {
  const wrap = document.createElement("div");
  wrap.className = role === "user" ? "flex justify-end" : "flex justify-start";

  const bubble = document.createElement("div");
  bubble.className =
    role === "user"
      ? "max-w-[80%] bg-slate-800 text-white rounded-2xl rounded-br-md px-4 py-2 shadow"
      : "max-w-[80%] bg-slate-200 text-slate-800 rounded-2xl rounded-bl-md px-4 py-2 shadow";
  bubble.textContent = content;
  wrap.appendChild(bubble);

  if (meta) {
    const metaEl = document.createElement("div");
    metaEl.className = "text-[10px] text-slate-400 mt-1 ml-1";
    metaEl.textContent = meta;
    wrap.appendChild(metaEl);
  }

  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function renderTyping() {
  const wrap = document.createElement("div");
  wrap.id = "typing";
  wrap.className = "flex justify-start";
  wrap.innerHTML =
    '<div class="bg-slate-200 rounded-2xl px-4 py-2 typing"><span></span><span></span><span></span></div>';
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function removeTyping() {
  const t = document.getElementById("typing");
  if (t) t.remove();
}

async function send(text) {
  history.push({ role: "user", content: text });
  renderMessage("user", text);
  renderTyping();
  sendBtn.disabled = true;
  statusEl.textContent = "Max is thinking…";

  const t0 = performance.now();
  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history, model: modelEl.value }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

    removeTyping();
    history.push({ role: "assistant", content: data.text });

    let meta = `${data.metrics.model} · ${data.metrics.tokens_per_second.toFixed(1)} t/s · ${elapsed}s`;
    if (data.tool_calls && data.tool_calls.length) {
      meta += ` · tool: ${data.tool_calls[0].name}`;
    }
    renderMessage("assistant", data.text, meta);
    statusEl.textContent = "";
  } catch (err) {
    removeTyping();
    renderMessage("assistant", `Sorry, something went wrong: ${err.message}`);
    statusEl.textContent = "";
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = "";
  send(text);
});

// Greet on load
window.addEventListener("DOMContentLoaded", () => {
  renderMessage(
    "assistant",
    "¡Hola! Soy Max, el asistente de Sharp & Co. ¿En qué te puedo ayudar hoy?"
  );
  inputEl.focus();
});
