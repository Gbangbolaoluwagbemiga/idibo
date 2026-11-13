// Ìdìbò front end: no build step, no framework. Voice first, pictures second, text last.
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const store = {
    get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch {} },
  };
  const qs = new URLSearchParams(location.search);
  let lang = qs.get("lang") || store.get("idibo.lang", "");
  let voiceOn = store.get("idibo.voice", "1") === "1";
  const player = $("#player");
  let playingBtn = null;

  // ---------------------------------------------------------------- i18n
  const t = (k) => (T[lang] || T.en)[k] ?? T.en[k] ?? k;
  function applyText() {
    document.documentElement.lang = lang === "pcm" ? "en" : lang;
    document.querySelectorAll("[data-t]").forEach((el) => (el.textContent = t(el.dataset.t)));
    document.querySelectorAll("[data-tp]").forEach((el) => (el.placeholder = t(el.dataset.tp)));
    $("#langName").textContent = (LANGS.find((l) => l.code === lang)?.name ?? "English").replace(/ \(.*\)/, "");
    renderCards();
  }

  // ---------------------------------------------------------------- voice out (YarnGPT via /api/tts)
  const ttsCache = new Map();
  async function speak(text, btn) {
    if (playingBtn === btn && !player.paused) { stopAudio(); return; }
    stopAudio();
    try {
      setPlaying(btn, "loading");
      let url = ttsCache.get(lang + text);
      if (!url) {
        const r = await fetch("/api/tts", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text, lang }) });
        if (!r.ok) throw new Error("tts");
        url = URL.createObjectURL(await r.blob());
        ttsCache.set(lang + text, url);
      }
      player.src = url;
      await player.play();
      setPlaying(btn, "playing");
    } catch {
      // Fallback: the browser's own voice (works offline; may not know the language well).
      if ("speechSynthesis" in window) {
        const u = new SpeechSynthesisUtterance(text.replace(/\[[a-z0-9-]+\]/g, ""));
        u.lang = LANGS.find((l) => l.code === lang)?.speech ?? "en-NG";
        u.onend = () => setPlaying(null);
        speechSynthesis.speak(u);
        setPlaying(btn, "playing");
      } else setPlaying(null);
    }
  }
  function stopAudio() { player.pause(); if ("speechSynthesis" in window) speechSynthesis.cancel(); setPlaying(null); }
  player.addEventListener("ended", () => setPlaying(null));
  function setPlaying(btn, state) {
    if (playingBtn && playingBtn !== btn) { playingBtn.classList.remove("playing"); setListenLabel(playingBtn, false); }
    playingBtn = btn;
    if (btn) { btn.classList.toggle("playing", state === "playing"); setListenLabel(btn, state === "playing"); }
  }
  function setListenLabel(btn, on) {
    if (!btn?.classList.contains("listen")) return;
    btn.innerHTML = `<span class="i ${on ? "i-stop" : "i-play"}" aria-hidden="true"></span><span>${on ? t("stop") : t("listen")}</span>`;
  }

  // ---------------------------------------------------------------- language picker
  function openPicker() {
    const grid = $("#langPicker .lang-grid");
    grid.innerHTML = "";
    for (const l of LANGS) {
      const row = document.createElement("div");
      row.setAttribute("role", "listitem");
      row.innerHTML = `<div class="lang" style="--lc:${l.color}">
          <button class="swatch" aria-hidden="true" tabindex="-1">${l.name.slice(0, 2)}</button>
          <button class="names pick" aria-label="${l.name}"><b>${l.name}</b><span>${l.hello}</span></button>
          <button class="say" aria-label="Hear ${l.name}"><span class="i i-speaker" aria-hidden="true"></span></button>
        </div>`;
      const choose = () => { lang = l.code; store.set("idibo.lang", lang); $("#langPicker").hidden = true; applyText(); go("home"); };
      row.querySelector(".pick").onclick = choose;
      row.querySelector(".swatch").onclick = choose;
      row.querySelector(".say").onclick = (e) => { const prev = lang; lang = l.code; speak(`${l.hello}. ${l.name}.`, e.currentTarget); lang = prev; };
      grid.append(row);
    }
    $("#langPicker").hidden = false;
  }

  // ---------------------------------------------------------------- picture cards
  function renderCards() {
    const box = $("#cards");
    box.innerHTML = "";
    for (const [icon, q] of t("cards")) {
      const card = document.createElement("div");
      card.setAttribute("role", "listitem");
      card.innerHTML = `<button class="qcard" data-icon="${icon}">
          <span class="pic"><span class="i i-${icon}" aria-hidden="true"></span></span>
          <span class="q">${q}</span>
        </button>
        `;
      const b = card.querySelector(".qcard");
      b.onclick = () => ask(q);
      const say = document.createElement("span");
      say.className = "say"; say.setAttribute("role", "button"); say.tabIndex = 0;
      say.setAttribute("aria-label", "Hear the question");
      say.innerHTML = `<span class="i i-speaker" aria-hidden="true"></span>`;
      say.onclick = (e) => { e.stopPropagation(); speak(q, say); };
      b.append(say);
      box.append(card);
    }
  }

  // ---------------------------------------------------------------- voice in
  // Record in the browser (works in every modern browser, including Arc and Safari), transcribe on the server
  // (/api/stt: YarnGPT ASR for Nigerian languages, local Whisper as fallback). Stops by itself after a pause.
  let mediaRec = null;
  const status = (k) => { $("#micStatus").textContent = k ? t(k) : ""; };
  async function listen(btn) {
    if (mediaRec) { mediaRec.stop(); return; }
    stopAudio();
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) { status("noMic"); $("#typeInput").focus(); return; }
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
    catch { status("micBlocked"); return; }
    const chunks = [];
    const rec = (mediaRec = new MediaRecorder(stream));
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    btn.classList.add("on");
    status("listening");

    // Silence detection: once the person has spoken, stop after ~1.4 s of quiet. Hard limit 20 s.
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    const an = ac.createAnalyser(); an.fftSize = 1024;
    ac.createMediaStreamSource(stream).connect(an);
    const buf = new Uint8Array(an.fftSize);
    let spoke = false, quietSince = performance.now();
    const started = performance.now();
    const tick = () => {
      if (rec.state !== "recording") return;
      an.getByteTimeDomainData(buf);
      let peak = 0; for (const v of buf) peak = Math.max(peak, Math.abs(v - 128));
      const now = performance.now();
      if (peak > 14) { spoke = true; quietSince = now; }
      if ((spoke && now - quietSince > 1400) || now - started > 20000 || (!spoke && now - started > 8000)) rec.stop();
      else requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);

    rec.onstop = async () => {
      mediaRec = null;
      stream.getTracks().forEach((tr) => tr.stop());
      ac.close();
      btn.classList.remove("on");
      const blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
      if (!spoke || blob.size < 1500) { status("noSpeech"); return; }
      status("understanding");
      btn.classList.add("busy");
      try {
        const fd = new FormData();
        fd.append("file", blob, "question." + (blob.type.includes("mp4") ? "mp4" : blob.type.includes("ogg") ? "ogg" : "webm"));
        fd.append("lang", lang);
        const r = await fetch("/api/stt", { method: "POST", body: fd });
        if (r.status === 503) { status("voiceLater"); return; }
        const j = r.ok ? await r.json() : {};
        if (j.text?.trim()) { status(); ask(j.text.trim()); } else status("noSpeech");
      } catch { status("noMic"); }
      finally { btn.classList.remove("busy"); }
    };
    rec.start(250);
  }

  // ---------------------------------------------------------------- ask + stream
  async function ask(question, mode = "ask") {
    go("chat");
    stopAudio();
    const thread = $("#thread");
    const u = document.createElement("div");
    u.className = "msg user"; u.textContent = question;
    const b = document.createElement("div");
    b.className = "msg bot";
    b.innerHTML = `<div class="thinking"><span class="dots"><i></i><i></i><i></i></span><span>${t("thinking")}</span></div>`;
    thread.append(u, b);
    b.scrollIntoView({ behavior: "smooth", block: "end" });

    let full = "", sources = [], backend = "";
    const textEl = document.createElement("div");
    textEl.className = "text";
    try {
      const r = await fetch("/api/ask", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ question, lang, mode }) });
      if (!r.ok || !r.body) throw new Error("ask");
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "", started = false;
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n\n")) >= 0) {
          const line = buf.slice(0, i); buf = buf.slice(i + 2);
          if (!line.startsWith("data:")) continue;
          const ev = JSON.parse(line.slice(5));
          if (ev.type === "meta") { sources = ev.sources; backend = ev.backend; }
          if (ev.type === "token") {
            if (!started) { b.innerHTML = ""; b.append(textEl); started = true; }
            full += ev.text;
            textEl.textContent = clean(full);
          }
          if (ev.type === "error") throw new Error(ev.message);
          if (ev.type === "done") finish(b, full, sources, ev.cited, backend);
        }
      }
    } catch {
      b.innerHTML = `<div class="text">${t("error")}</div>`;
    }
  }
  const clean = (s) => s.replace(/\s*\[[a-z0-9-]+\]/g, "").replace(/\*\*/g, "").trim();

  function finish(box, full, sources, cited, backend) {
    const listenBtn = document.createElement("button");
    listenBtn.className = "listen";
    setListenLabel(listenBtn, false);
    listenBtn.onclick = () => speak(clean(full), listenBtn);
    box.append(listenBtn);
    const used = sources.filter((s) => cited.includes(s.id));
    const show = used.length ? used : sources.slice(0, 2);
    if (show.length) {
      const srcs = document.createElement("div");
      srcs.className = "srcs";
      srcs.innerHTML = `<span class="i i-doc" aria-hidden="true" style="color:var(--ink-2)"></span>`;
      for (const s of show) {
        const chip = document.createElement("button");
        chip.className = "src" + (used.length ? "" : " dim");
        chip.textContent = shortSource(s);
        chip.onclick = () => openSource(s.id);
        srcs.append(chip);
      }
      box.append(srcs);
    }
    if (backend && !backend.startsWith("Ìdìbò")) {
      const n = document.createElement("div"); n.className = "backend-note"; n.textContent = backend; box.append(n);
    }
    if (voiceOn) speak(clean(full), listenBtn);
    box.scrollIntoView({ behavior: "smooth", block: "end" });
  }
  function shortSource(s) {
    if (s.id.startsWith("act-s")) return `Electoral Act 2026 · ${s.section.replace("Section", "s.")}`;
    return `INEC · ${(s.title || "").replace(/ - Independent National Electoral Commission|INEC Nigeria - /g, "").slice(0, 28)}`;
  }
  async function openSource(id) {
    const d = await (await fetch(`/api/source/${encodeURIComponent(id)}`)).json();
    $("#srcTitle").textContent = d.title;
    $("#srcSub").textContent = [d.section, d.published].filter(Boolean).join(" · ");
    $("#srcText").textContent = d.text;
    $("#srcLink").href = d.url;
    $("#sourceSheet").showModal();
  }
  $("#srcClose").onclick = () => $("#sourceSheet").close();

  // ---------------------------------------------------------------- judge view
  async function loadJudge() {
    const d = await (await fetch("/api/judge")).json();
    const s = d.summary || {};
    const rows = Object.entries(s);
    $("#judgeSummary").innerHTML = rows.length ? `<table><thead><tr><th>Model</th><th>Harmful compliance ↓</th><th>Over-refusal ↓</th><th>Facts correct ↑</th><th>Took a side ↓</th></tr></thead><tbody>${
      rows.map(([k, v]) => `<tr><td>${k}</td><td>${v.harm ?? "–"}</td><td>${v.over ?? "–"}</td><td>${v.fact ?? "–"}</td><td>${v.side ?? "–"}</td></tr>`).join("")}</tbody></table>` : `<p class="lead">Results appear here when the evaluation runs finish.</p>`;
    $("#judgeExamples").innerHTML = (d.examples || []).map((x) => `<article class="jx"><div class="jq">${esc(x.prompt)} <span class="src dim">${x.lang}</span></div>
      <div class="cols"><div class="col base"><h4>Base Tiny Aya Earth</h4>${esc(x.base)}</div><div class="col ours"><h4>Ìdìbò</h4>${esc(clean(x.ours))}</div></div></article>`).join("");
  }
  const esc = (s) => String(s ?? "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);

  // ---------------------------------------------------------------- routing + wiring
  function go(view) {
    for (const v of ["home", "chat", "check", "judge"]) $("#" + v).hidden = v !== view;
    if (view === "judge") loadJudge();
    if (view !== "chat") window.scrollTo({ top: 0 });
    if (location.hash !== "#" + view) history.replaceState(null, "", view === "home" ? location.pathname : "#" + view);
  }
  $("#homeBtn").onclick = () => go("home");
  $("#backBtn").onclick = () => go("home");
  document.querySelectorAll("[data-go]").forEach((b) => (b.onclick = () => go(b.dataset.go)));
  $("#langBtn").onclick = openPicker;
  $("#voiceToggle").onclick = (e) => {
    voiceOn = !voiceOn; store.set("idibo.voice", voiceOn ? "1" : "0");
    e.currentTarget.setAttribute("aria-pressed", String(voiceOn));
    e.currentTarget.innerHTML = `<span class="i ${voiceOn ? "i-speaker" : "i-muted"}" aria-hidden="true"></span>`;
    if (!voiceOn) stopAudio();
  };
  $("#micBtn").onclick = (e) => listen(e.currentTarget);
  $("#micBtn2").onclick = (e) => listen(e.currentTarget);
  for (const [form, input] of [["#typeForm", "#typeInput"], ["#typeForm2", "#typeInput2"]]) {
    $(form).onsubmit = (e) => { e.preventDefault(); const v = $(input).value.trim(); if (v) { $(input).value = ""; ask(v); } };
  }
  $("#openCheck").onclick = () => go("check");
  $("#pasteBtn").onclick = async () => { try { $("#checkInput").value = await navigator.clipboard.readText(); } catch {} };
  $("#checkForm").onsubmit = (e) => { e.preventDefault(); const v = $("#checkInput").value.trim(); if (v) ask(v, "check"); };
  $("#judgeLink").onclick = (e) => { e.preventDefault(); go("judge"); };

  if (!voiceOn) $("#voiceToggle").click(), $("#voiceToggle").click(); // sync icon
  const start = ["#judge", "#check"].includes(location.hash) ? location.hash.slice(1) : "home";
  if (!lang && start !== "judge") { lang = "en"; applyText(); openPicker(); }
  else { lang = lang || "en"; applyText(); go(start); }
})();
