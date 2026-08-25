const STORAGE_KEY = "atlas-travel-session-v1";

const elements = {
  form: document.querySelector("#chatForm"),
  input: document.querySelector("#messageInput"),
  send: document.querySelector("#sendButton"),
  messages: document.querySelector("#messages"),
  welcome: document.querySelector("#welcome"),
  title: document.querySelector("#conversationTitle"),
  tripDestination: document.querySelector("#tripDestination"),
  tripLabel: document.querySelector("#tripLabel"),
  route: document.querySelector("#routeFact"),
  duration: document.querySelector("#durationFact"),
  budget: document.querySelector("#budgetFact"),
  sources: document.querySelector("#sourceList"),
  sourceCount: document.querySelector("#sourceCount"),
  briefStatus: document.querySelector("#briefStatus"),
  charCount: document.querySelector("#charCount"),
  toast: document.querySelector("#toast"),
  sidebar: document.querySelector("#sidebar"),
  countries: document.querySelector("#countries"),
  activities: document.querySelector("#activities"),
  minDays: document.querySelector("#minDays"),
  maxBudget: document.querySelector("#maxBudget"),
};

let state = loadState();
let busy = false;

function emptyState() {
  return { messages: [], trip: {}, sources: [], preferences: {} };
}

function loadState() {
  try {
    return { ...emptyState(), ...JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}") };
  } catch {
    return emptyState();
  }
}

function saveState() {
  state.preferences = {
    countries: elements.countries.value,
    activities: elements.activities.value,
    minDays: elements.minDays.value,
    maxBudget: elements.maxBudget.value,
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function restorePreferences() {
  const preferences = state.preferences || {};
  elements.countries.value = preferences.countries || "";
  elements.activities.value = preferences.activities || "";
  elements.minDays.value = preferences.minDays || "";
  elements.maxBudget.value = preferences.maxBudget || "";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function markdown(value) {
  const safe = escapeHtml(value);
  const lines = safe.split("\n");
  let list = null;
  const output = [];
  const closeList = () => {
    if (list) output.push(`</${list}>`);
    list = null;
  };
  for (const line of lines) {
    const heading = line.match(/^(#{2,3})\s+(.+)/);
    const bullet = line.match(/^[-*]\s+(.+)/);
    const ordered = line.match(/^\d+[.)]\s+(.+)/);
    if (heading) {
      closeList();
      output.push(`<h${heading[1].length}>${inlineMarkdown(heading[2])}</h${heading[1].length}>`);
    } else if (bullet || ordered) {
      const type = bullet ? "ul" : "ol";
      if (list !== type) { closeList(); list = type; output.push(`<${type}>`); }
      output.push(`<li>${inlineMarkdown((bullet || ordered)[1])}</li>`);
    } else if (line.trim() === "---") {
      closeList();
    } else if (line.trim()) {
      closeList();
      output.push(`<p>${inlineMarkdown(line)}</p>`);
    }
  }
  closeList();
  return output.join("");
}

function inlineMarkdown(value) {
  return value
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/&lt;(https?:\/\/[^&]+)&gt;/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');
}

function renderMessage(message) {
  const article = document.createElement("article");
  article.className = `message ${message.role}`;
  article.innerHTML = `
    <div class="avatar">${message.role === "user" ? "YO" : "A"}</div>
    <div class="message-body">
      <div class="message-meta">${message.role === "user" ? "You" : "Atlas"}</div>
      ${message.role === "assistant" ? markdown(message.content) : `<p>${escapeHtml(message.content)}</p>`}
    </div>`;
  elements.messages.appendChild(article);
}

function renderConversation() {
  elements.messages.querySelectorAll(".message").forEach(node => node.remove());
  elements.welcome.hidden = state.messages.length > 0;
  state.messages.forEach(renderMessage);
  const firstUser = state.messages.find(message => message.role === "user");
  elements.title.textContent = firstUser ? firstUser.content.slice(0, 48) : "Plan a new journey";
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function renderTrip() {
  const trip = state.trip || {};
  const destination = trip.destination || "Open canvas";
  elements.tripDestination.textContent = destination;
  elements.tripLabel.textContent = trip.destination ? "Current destination" : "Your next journey";
  elements.route.textContent = trip.origin && trip.destination ? `${trip.origin} → ${trip.destination}` : trip.destination || "Not set";
  elements.duration.textContent = trip.duration ? `${trip.duration} days` : "Flexible";
  elements.budget.textContent = trip.budget ? `$${Number(trip.budget).toLocaleString()}` : "Not set";
  elements.briefStatus.textContent = trip.destination ? "In progress" : "Waiting for details";
}

function renderSources() {
  const sources = state.sources || [];
  elements.sourceCount.textContent = `${sources.length} matched`;
  if (!sources.length) {
    elements.sources.innerHTML = '<div class="empty-sources">Relevant guides will appear here as Atlas plans.</div>';
    return;
  }
  elements.sources.innerHTML = sources.map(source => {
    const title = escapeHtml(source.title || "Travel source");
    const heading = source.url
      ? `<a href="${escapeHtml(source.url)}" target="_blank" rel="noopener">${title}</a>`
      : `<strong>${title}</strong>`;
    return `<article class="source-item">${heading}<p>${escapeHtml(source.excerpt || "")}</p><span class="source-score">${Math.round(source.score * 100)}% relevance</span></article>`;
  }).join("");
}

function showTyping() {
  const article = document.createElement("article");
  article.id = "typingMessage";
  article.className = "message assistant";
  article.innerHTML = '<div class="avatar">A</div><div class="message-body"><div class="message-meta">Atlas is planning</div><div class="typing"><i></i><i></i><i></i></div></div>';
  elements.messages.appendChild(article);
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function setBusy(value) {
  busy = value;
  elements.send.disabled = value;
  elements.input.disabled = value;
}

function csv(value) {
  return value.split(",").map(item => item.trim()).filter(Boolean);
}

async function sendMessage(text) {
  if (busy || text.trim().length < 2) return;
  const message = text.trim();
  const history = state.messages.slice(-8);
  state.messages.push({ role: "user", content: message });
  elements.input.value = "";
  resizeComposer();
  saveState();
  renderConversation();
  showTyping();
  setBusy(true);

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        history,
        filters: {
          countries: csv(elements.countries.value),
          activities: csv(elements.activities.value),
          min_days: Number(elements.minDays.value) || null,
          max_budget: Number(elements.maxBudget.value) || null,
        },
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Planning failed");
    state.messages.push({ role: "assistant", content: payload.answer });
    state.trip = payload.trip || {};
    state.sources = payload.sources || [];
    saveState();
    renderConversation();
    renderTrip();
    renderSources();
  } catch (error) {
    state.messages.push({ role: "assistant", content: `I couldn't complete that plan. ${error.message}` });
    saveState();
    renderConversation();
    toast("Atlas could not reach the planning service.");
  } finally {
    document.querySelector("#typingMessage")?.remove();
    setBusy(false);
    elements.input.focus();
  }
}

function resizeComposer() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 140)}px`;
  elements.charCount.textContent = `${elements.input.value.length} / 4000`;
}

function toast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  window.setTimeout(() => elements.toast.classList.remove("show"), 2600);
}

function resetTrip() {
  state = emptyState();
  localStorage.removeItem(STORAGE_KEY);
  restorePreferences();
  renderConversation();
  renderTrip();
  renderSources();
  elements.input.focus();
  toast("New trip ready.");
}

function exportPlan() {
  if (!state.messages.length) return toast("There is no itinerary to export yet.");
  const content = state.messages.map(message => `## ${message.role === "user" ? "You" : "Atlas"}\n\n${message.content}`).join("\n\n");
  const blob = new Blob([content], { type: "text/markdown" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "atlas-travel-plan.md";
  link.click();
  URL.revokeObjectURL(link.href);
}

elements.form.addEventListener("submit", event => { event.preventDefault(); sendMessage(elements.input.value); });
elements.input.addEventListener("input", resizeComposer);
elements.input.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendMessage(elements.input.value); }
});
document.querySelectorAll("[data-prompt]").forEach(button => button.addEventListener("click", () => sendMessage(button.dataset.prompt)));
document.querySelector("#newTripButton").addEventListener("click", resetTrip);
document.querySelector("#exportButton").addEventListener("click", exportPlan);
document.querySelector("#menuButton").addEventListener("click", () => elements.sidebar.classList.toggle("open"));
[elements.countries, elements.activities, elements.minDays, elements.maxBudget].forEach(input => input.addEventListener("change", saveState));

restorePreferences();
renderConversation();
renderTrip();
renderSources();
resizeComposer();
