(function () {
  const REVEAL_DELAY = 0.35; // seconds into an event before the label "resolves"

  const STATUS_LABEL = {
    covered: "คลุมผ้าใบ",
    uncovered: "ไม่คลุมผ้าใบ",
    untidy: "คลุมผ้าใบไม่เรียบร้อย",
  };
  const STATUS_BADGE_CLASS = {
    covered: "badge-covered",
    uncovered: "badge-uncovered",
    untidy: "badge-untidy",
  };

  const video = document.getElementById("video");
  const canvas = document.getElementById("overlay");
  const ctx = canvas.getContext("2d");
  const videoWrap = document.getElementById("video-wrap");
  const historyList = document.getElementById("history-list");
  const historyPlaceholder = document.getElementById("history-placeholder");
  const filterHint = document.getElementById("filter-hint");
  const statButtons = Array.from(document.querySelectorAll(".stat-btn"));

  let videoSize = [1280, 720];
  let events = [];
  let countedIds = new Set();
  let historyEntries = [];
  let counts = { covered: 0, uncovered: 0, untidy: 0 };
  let filterStatus = null;
  let rafHandle = null;

  function fetchDetections() {
    fetch("/api/detections")
      .then((r) => r.json())
      .then((data) => {
        videoSize = data.video_size;
        events = data.events;
      })
      .catch((err) => console.error("failed to load detections", err));
  }

  function resizeCanvas() {
    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    canvas.style.width = rect.width + "px";
    canvas.style.height = rect.height + "px";
  }

  function scaleBox(box) {
    const sx = canvas.width / videoSize[0];
    const sy = canvas.height / videoSize[1];
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy];
  }

  function drawBracketBox(x, y, w, h, color) {
    const len = Math.max(10, Math.min(w, h) * 0.2);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.25;
    ctx.strokeRect(x, y, w, h);
    ctx.globalAlpha = 1;
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;
    const corners = [
      [x, y, 1, 1],
      [x + w, y, -1, 1],
      [x, y + h, 1, -1],
      [x + w, y + h, -1, -1],
    ];
    for (const [cx, cy, dx, dy] of corners) {
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + dx * len, cy);
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx, cy + dy * len);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawLabel(x, y, text, color) {
    ctx.font = "600 12px 'Consolas', 'Leelawadee UI', 'Tahoma', sans-serif";
    const padX = 8;
    const w = ctx.measureText(text).width + padX * 2;
    const h = 22;
    let ly = y - h - 6;
    if (ly < 0) ly = y + 6;
    ctx.fillStyle = "rgba(6, 10, 19, 0.82)";
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    roundRect(x, ly, w, h, 5);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.fillText(text, x + padX, ly + 15);
  }

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  const STATUS_COLOR = { covered: "#34d399", uncovered: "#f43f5e", untidy: "#fbbf24" };
  const SCAN_COLOR = "#22d3ee";

  function drawFrame() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!events.length || !canvas.width) return;

    const t = video.currentTime;
    for (const ev of events) {
      if (t < ev.t_start || t > ev.t_end) continue;
      const revealed = t - ev.t_start >= REVEAL_DELAY;

      const [cx, cy, cw, ch] = scaleBox(ev.boxes.cargo);
      const [px, py, pw, ph] = scaleBox(ev.boxes.plate);
      const color = revealed ? STATUS_COLOR[ev.status] : SCAN_COLOR;

      drawBracketBox(cx, cy, cw, ch, color);
      drawBracketBox(px, py, pw, ph, color);

      if (!revealed) {
        drawLabel(cx, cy, "SCANNING TARP...", SCAN_COLOR);
        drawLabel(px, py, "READING PLATE...", SCAN_COLOR);
      } else {
        drawLabel(cx, cy, STATUS_LABEL[ev.status], STATUS_COLOR[ev.status]);
        drawLabel(px, py, ev.plate_text, "#e7ecf5");

        if (!countedIds.has(ev.id)) {
          countedIds.add(ev.id);
          registerDetection(ev);
        }
      }
    }
  }

  function loop() {
    drawFrame();
    rafHandle = requestAnimationFrame(loop);
  }

  function registerDetection(ev) {
    counts[ev.status] += 1;
    updateCounts();
    historyEntries.unshift(ev);
    renderHistory();
  }

  function updateCounts() {
    document.getElementById("count-covered").textContent = counts.covered;
    document.getElementById("count-uncovered").textContent = counts.uncovered;
    document.getElementById("count-untidy").textContent = counts.untidy;
  }

  function renderHistory() {
    const visible = filterStatus
      ? historyEntries.filter((e) => e.status === filterStatus)
      : historyEntries;

    historyPlaceholder.style.display = historyEntries.length ? "none" : "flex";
    historyList.querySelectorAll(".history-row").forEach((el) => el.remove());

    for (const ev of visible) {
      const row = document.createElement("div");
      row.className = "history-row";
      row.innerHTML = `
        <img src="/static/${ev.thumb}" alt="truck">
        <div class="history-info">
          <div class="plate">${ev.plate_text}</div>
          <div class="meta">${ev.province} · ${ev.timestamp}</div>
        </div>
        <div class="badge ${STATUS_BADGE_CLASS[ev.status]}">${STATUS_LABEL[ev.status]}</div>
      `;
      historyList.appendChild(row);
    }
  }

  statButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const status = btn.dataset.status;
      if (filterStatus === status) {
        filterStatus = null;
        filterHint.textContent = "";
        statButtons.forEach((b) => b.classList.remove("active"));
      } else {
        filterStatus = status;
        filterHint.textContent = `กำลังกรอง: ${STATUS_LABEL[status]} (กดปุ่มเดิมอีกครั้งเพื่อยกเลิก)`;
        statButtons.forEach((b) => b.classList.toggle("active", b === btn));
      }
      renderHistory();
    });
  });

  window.addEventListener("resize", resizeCanvas);
  video.addEventListener("loadedmetadata", () => {
    resizeCanvas();
    video.playbackRate = 0.5;
  });

  fetchDetections();
  resizeCanvas();
  updateCounts();
  loop();
  video.playbackRate = 0.5;
  video.play().catch(() => {});
})();
