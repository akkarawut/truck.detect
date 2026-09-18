(function () {
  const STATUS_LABEL = {
    covered: "คลุมผ้าใบ",
    uncovered: "ไม่คลุมผ้าใบ",
    untidy: "คลุมผ้าใบไม่เรียบร้อย",
  };
  const STATUS_COLOR = { covered: "#2e7d32", uncovered: "#c62828", untidy: "#d98e04" };
  const STATUS_TEXT_COLOR = { covered: "#fff", uncovered: "#fff", untidy: "#1f2328" };
  const PENDING_COLOR = "#ffffff";

  const video = document.getElementById("video");
  const canvas = document.getElementById("overlay");
  const ctx = canvas.getContext("2d");
  const historyList = document.getElementById("history-list");
  const historyPlaceholder = document.getElementById("history-placeholder");
  const logTotal = document.getElementById("log-total");
  const filterHint = document.getElementById("filter-hint");
  const clock = document.getElementById("clock");
  const summaryButtons = Array.from(document.querySelectorAll(".summary-item"));

  let videoSize = [1920, 1080];
  let fps = 24;
  let events = [];
  let countedIds = new Set();
  let historyEntries = [];
  let counts = { covered: 0, uncovered: 0, untidy: 0 };
  let filterStatus = null;
  let viewW = 0;
  let viewH = 0;

  function fetchDetections() {
    fetch("/api/detections")
      .then((r) => r.json())
      .then((data) => {
        videoSize = data.video_size;
        fps = data.fps;
        events = data.events;
      })
      .catch((err) => console.error("failed to load detections", err));
  }

  function resizeCanvas() {
    const rect = video.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    viewW = rect.width;
    viewH = rect.height;
    canvas.width = Math.round(viewW * dpr);
    canvas.height = Math.round(viewH * dpr);
    canvas.style.width = viewW + "px";
    canvas.style.height = viewH + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // Box for this event at a fractional frame, interpolated between the
  // per-frame boxes from the detector so it glides with the video.
  function boxAt(ev, frame) {
    const i = frame - ev.first_frame;
    if (i < 0 || i > ev.boxes.length - 1) return null;
    const i0 = Math.floor(i);
    const i1 = Math.min(i0 + 1, ev.boxes.length - 1);
    const k = i - i0;
    const a = ev.boxes[i0];
    const b = ev.boxes[i1];
    const sx = viewW / videoSize[0];
    const sy = viewH / videoSize[1];
    return [
      (a[0] + (b[0] - a[0]) * k) * sx,
      (a[1] + (b[1] - a[1]) * k) * sy,
      (a[2] + (b[2] - a[2]) * k) * sx,
      (a[3] + (b[3] - a[3]) * k) * sy,
    ];
  }

  function drawBox(x, y, w, h, color) {
    ctx.save();
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.strokeRect(x + 1, y + 1, w - 2, h - 2);
    ctx.restore();
  }

  function drawTag(x, y, lines, bg, fg) {
    ctx.save();
    ctx.font = "500 13px 'IBM Plex Sans Thai', 'Leelawadee UI', Tahoma, sans-serif";
    ctx.textBaseline = "middle";
    const padX = 7;
    const lineH = 20;
    const w = Math.max(...lines.map((s) => ctx.measureText(s).width)) + padX * 2;
    const h = lineH * lines.length;
    let tx = Math.max(0, Math.min(x, viewW - w));
    let ty = y - h;
    if (ty < 0) ty = y; // no room above: tuck the tag inside the box
    ctx.fillStyle = bg;
    ctx.fillRect(tx, ty, w, h);
    ctx.fillStyle = fg;
    lines.forEach((s, i) => ctx.fillText(s, tx + padX, ty + lineH * i + lineH / 2 + 1));
    ctx.restore();
  }

  function drawFrame() {
    ctx.clearRect(0, 0, viewW, viewH);
    if (!events.length || !viewW) return;

    const frame = video.currentTime * fps;
    for (const ev of events) {
      const box = boxAt(ev, frame);
      if (!box) continue;
      const [x, y, w, h] = box;
      const logged = frame >= ev.log_frame;

      if (logged) {
        const color = STATUS_COLOR[ev.status];
        drawBox(x, y, w, h, color);
        drawTag(x, y, [`${ev.plate_text}  ·  ${STATUS_LABEL[ev.status]}`], color, STATUS_TEXT_COLOR[ev.status]);
        if (!countedIds.has(ev.id)) {
          countedIds.add(ev.id);
          registerDetection(ev);
        }
      } else {
        drawBox(x, y, w, h, PENDING_COLOR);
        drawTag(x, y, [`รถบรรทุก #${ev.id}`], "rgba(0, 0, 0, 0.75)", "#fff");
      }
    }
  }

  function loop() {
    drawFrame();
    requestAnimationFrame(loop);
  }

  function registerDetection(ev) {
    counts[ev.status] += 1;
    updateCounts();
    historyEntries.unshift(ev);
    renderHistory(ev.id);
  }

  function updateCounts() {
    document.getElementById("count-covered").textContent = counts.covered;
    document.getElementById("count-uncovered").textContent = counts.uncovered;
    document.getElementById("count-untidy").textContent = counts.untidy;
    logTotal.textContent = `${historyEntries.length} คัน`;
  }

  function renderHistory(newId) {
    const visible = filterStatus
      ? historyEntries.filter((e) => e.status === filterStatus)
      : historyEntries;

    historyList.querySelectorAll(".log-row").forEach((el) => el.remove());
    historyPlaceholder.style.display = visible.length ? "none" : "block";
    historyPlaceholder.textContent = historyEntries.length ? "ไม่มีรายการในหมวดนี้" : "ยังไม่มีรถผ่าน";
    logTotal.textContent = `${historyEntries.length} คัน`;

    for (const ev of visible) {
      const row = document.createElement("div");
      row.className = "log-row" + (ev.id === newId ? " is-new" : "");
      row.innerHTML = `
        <img src="/static/${ev.thumb}" alt="">
        <div class="log-info">
          <div class="log-plate">${ev.plate_text}<span class="log-province">${ev.province}</span></div>
          <div class="log-time">${ev.timestamp}</div>
          <div class="log-status ${ev.status}"><i class="dot dot-${ev.status}"></i>${STATUS_LABEL[ev.status]}</div>
        </div>
      `;
      historyList.appendChild(row);
    }
  }

  summaryButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const status = btn.dataset.status;
      filterStatus = filterStatus === status ? null : status;
      summaryButtons.forEach((b) => b.classList.toggle("active", b.dataset.status === filterStatus));
      filterHint.textContent = filterStatus
        ? `แสดงเฉพาะ: ${STATUS_LABEL[filterStatus]} (กดอีกครั้งเพื่อแสดงทั้งหมด)`
        : "";
      renderHistory();
    });
  });

  video.addEventListener("loadedmetadata", resizeCanvas);
  new ResizeObserver(resizeCanvas).observe(video);

  function tickClock() {
    clock.textContent = new Date().toLocaleString("en-GB", {
      day: "2-digit", month: "2-digit", year: "numeric",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
  }
  setInterval(tickClock, 1000);
  tickClock();

  fetchDetections();
  resizeCanvas();
  updateCounts();
  loop();
  video.play().catch(() => {});
})();
