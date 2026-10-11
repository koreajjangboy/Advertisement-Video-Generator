const $episodes = document.getElementById("episodes");
const $form = document.getElementById("episode-form");

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.status === 204 ? null : res.json();
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

let appStatus = null; // { api_mode, llm: {...}, tts: { provider, fallback, active }, render: {...}, video_ai, host }

function providerBadge(p) {
  if (!p) return "";
  const tts = appStatus?.tts;
  const fallback = tts && p === tts.fallback && p !== tts.provider;
  return `<span class="badge tts-${p}" title="${fallback ? "기본 엔진 한도 초과로 폴백됨" : "TTS 엔진"}">${p}${fallback ? " (폴백)" : ""}</span>`;
}

async function loadStatus() {
  try {
    appStatus = await api("/api/status");
    const { api_mode, llm, tts, render } = appStatus;
    const ttsResting = tts.active !== tts.provider;
    const llmLimited = llm.provider !== "mock" && (llm.active === "rate-limited" || llm.active === "mock");
    const el = document.getElementById("tts-status");
    el.textContent = [
      `모드: ${api_mode}`,
      `대본: ${llm.active}${llmLimited ? " (한도 초과)" : ""}`,
      `TTS: ${tts.provider}${tts.fallback && tts.fallback !== tts.provider ? ` → 폴백 ${tts.fallback}` : ""}` +
        (ttsResting ? ` · 현재 ${tts.active} 사용 중 (한도 초과)` : ""),
      render.ffmpeg ? "FFmpeg 사용 가능" : "FFmpeg 없음",
      appStatus.video_ai?.enabled ? `AI 씬 영상: Luma ${appStatus.video_ai.model}` : "AI 씬 영상: 꺼짐",
    ].join(" · ");
    el.title = appStatus.video_ai?.reason ? `AI 씬 영상(Luma) 꺼짐: ${appStatus.video_ai.reason}` : "";
    el.classList.toggle("warn", ttsResting || llmLimited || !render.ffmpeg);
  } catch { /* 상태 표시는 선택 사항 */ }
}

const scriptsById = new Map(); // 편집기에서 원본 값을 꺼내기 위한 캐시
const episodeOfScript = new Map(); // script id → 에피소드 (여자주인공 설정 확인용)

const STATUS_LABEL = {
  draft: "초안", scripting: "대본 생성 중", scripted: "대본 준비", voicing: "음성 생성 중",
  clipping: "씬 영상 생성 중", clipped: "씬 영상 준비", rendering: "렌더링 중", done: "완료", failed: "실패",
};
const BUSY = ["scripting", "voicing", "clipping", "rendering"];

// ---------- 호스트 캐릭터 포즈 ----------
function hostPoses() {
  return appStatus?.host?.mode === "character" ? appStatus.host.poses : [];
}

// 씬에서 캐릭터가 연기할 감정 비트 (예: 공감·회상 → 놀람): 비트마다 포즈 썸네일 + 감정
function poseBadge(s) {
  const poses = hostPoses();
  const beats = (s.resolved_acting || [{ pose: s.resolved_pose }])
    .map((b) => ({ ...b, info: poses.find((p) => p.name === b.pose) }))
    .filter((b) => b.info);
  if (!beats.length) return "";
  const how = s.host_pose ? "지정" : "자동";
  const parts = beats.map((b) =>
    `${b.info.image_url ? `<img src="${b.info.image_url}" alt="">` : ""}${escapeHtml(b.label || b.info.label)}`);
  const title = beats.map((b) => `${b.label || ""} ${b.info.label} 포즈`.trim()).join(" → ");
  return `<span class="pose-badge" title="호스트 연기: ${escapeHtml(title)} (${how})">
    ${parts.join('<span class="beat-arrow">→</span>')}<small>${how}</small></span>`;
}

function poseSelect(s) {
  const poses = hostPoses();
  if (!poses.length) return "";
  const options = poses.map((p) =>
    `<option value="${p.name}" ${p.name === s.host_pose ? "selected" : ""}>${escapeHtml(p.label)}</option>`).join("");
  return `<label>호스트 포즈 <select name="host_pose">
    <option value="" ${s.host_pose ? "" : "selected"}>자동 (대본에 맞게)</option>${options}</select></label>`;
}

// ---------- 단계 표시 ----------
function stepProgress(ep) {
  const total = ep.scripts.length;
  const voiced = ep.scripts.filter((s) => s.audio_path).length;
  const clips = clipSummary(ep);
  const videoFresh = ep.video_path && ep.status === "done";
  return [
    { n: 1, title: "대본", done: total > 0 && ep.status !== "scripting", active: ep.status === "scripting",
      detail: total ? `${total}개 씬` : "없음" },
    { n: 2, title: "음성", done: total > 0 && voiced === total, active: ep.status === "voicing",
      detail: `${voiced}/${total}` },
    { n: 3, title: "씬 영상", done: total > 0 && clips.ready === total, active: ep.status === "clipping",
      detail: `${clips.ready}/${total}` },
    { n: 4, title: "최종 영상", done: Boolean(videoFresh), active: ep.status === "rendering",
      detail: videoFresh ? "완료" : ep.video_path ? "다시 렌더링 필요" : "없음" },
  ];
}

function stepNav(ep) {
  return `<ol class="step-nav">${stepProgress(ep).map((s) => `
    <li class="${s.active ? "active" : s.done ? "done" : ""}"><b>${s.n}단계</b> ${s.title}
      <span class="meta">${s.active ? "진행 중" : escapeHtml(s.detail)}</span></li>`).join("")}</ol>`;
}

// ---------- 1단계: 대본 생성 및 편집 ----------
function scriptStep(ep, busy) {
  const scripting = ep.status === "scripting";
  const scenes = ep.scripts.map((s) => renderScene(s, busy, ep)).join("");
  return `
    <section class="step ${scripting ? "active" : ""}">
      <div class="stage-title">1단계 · 대본 생성 및 편집
        <span class="meta">씬별 내레이션·화면 자막·영상 프롬프트를 확인하고 수정·저장</span></div>
      ${scripting ? `<div class="progress indeterminate"><div></div></div><div class="meta">대본 생성 중…</div>` : ""}
      ${heroineSettings(ep, busy)}
      ${scenes ? `<ol class="scenes">${scenes}</ol>` : ""}
      <div class="stage-actions">
        <button data-action="script" data-id="${ep.id}" ${busy ? "disabled" : ""}>
          ${scripting ? "생성 중…" : ep.scripts.length ? "대본 다시 생성" : "1단계: 대본 생성하기"}</button>
      </div>
    </section>`;
}

// ---------- 여자주인공 ----------
function heroineSettings(ep, busy) {
  const status = ep.heroine_enabled
    ? `등장 · ${escapeHtml(ep.heroine_dir)} · 대표 이미지 ${escapeHtml(ep.heroine_ref || "(첫 이미지)")} · ${ep.scripts.filter((s) => s.heroine).length}개 씬`
    : "등장 안 함";
  return `
    <div class="heroine-summary meta">여자주인공: ${status}
      <button type="button" class="secondary small" data-action="edit-heroine" ${busy ? "disabled" : ""}>설정 변경</button></div>
    <form class="heroine-fields heroine-edit" data-heroine data-episode-id="${ep.id}" hidden>
      <label class="inline"><input type="checkbox" name="heroine_enabled" data-role="heroine-toggle" ${ep.heroine_enabled ? "checked" : ""}> 여자주인공 등장</label>
      <div class="heroine-config">
        <label>캐릭터 폴더 <select data-role="heroine-folder" disabled><option>불러오는 중…</option></select></label>
        <label class="heroine-custom" hidden>폴더 경로 직접 입력 <span class="meta">(media 폴더 안 — 상대 경로 또는 절대 경로)</span>
          <input name="heroine_dir" value="${escapeHtml(ep.heroine_dir || appStatus?.heroine?.default_dir || "")}" disabled></label>
        <button type="button" class="secondary small" data-action="load-heroine" disabled>이미지 불러오기</button>
        <div class="heroine-images" data-selected="${escapeHtml(ep.heroine_ref)}"></div>
      </div>
      <div class="meta">켜면 등장 씬이 없을 때 본문 씬에 등장시키고, 씬별 등장 여부는 각 씬의 '수정'에서 바꿀 수 있습니다.</div>
      <div class="actions">
        <button type="submit">저장</button>
        <button type="button" class="secondary" data-action="cancel-heroine">취소</button>
      </div>
    </form>`;
}

const CUSTOM_FOLDER = "__custom__";

// 캐릭터 폴더 드롭다운: 서버가 스캔한 media/images 하위 폴더 + '직접 입력…'. 지금 값(heroine_dir)이 목록에 있으면 선택.
async function fillHeroineFolders(box) {
  const select = box.querySelector('[data-role="heroine-folder"]');
  const input = box.querySelector('[name="heroine_dir"]');
  try {
    const res = await api("/api/heroine/folders");
    const current = input.value.trim() || res.default_dir;
    select.innerHTML = res.folders.map((f) =>
      `<option value="${escapeHtml(f.dir)}">${escapeHtml(f.name)} (${f.count}장)</option>`).join("")
      + `<option value="${CUSTOM_FOLDER}">직접 입력…</option>`;
    const known = res.folders.some((f) => f.dir === current);
    select.value = known ? current : CUSTOM_FOLDER;
    input.value = known ? current : input.value.trim();
    box.querySelector(".heroine-custom").hidden = known;
  } catch (err) {
    select.innerHTML = `<option value="${CUSTOM_FOLDER}">직접 입력… (폴더 목록을 불러오지 못함)</option>`;
    box.querySelector(".heroine-custom").hidden = false;
  }
  box.dataset.foldersLoaded = "1";
}

// 체크박스에 따라 폴더 선택·입력·이미지 고르기를 켜고 끔
function setHeroineEnabled(box, enabled) {
  const config = box.querySelector(".heroine-config");
  config.classList.toggle("disabled", !enabled);
  config.querySelectorAll("select, input, button").forEach((el) => { el.disabled = !enabled; });
}

async function initHeroineBox(box) {
  const enabled = box.querySelector('[name="heroine_enabled"]').checked;
  setHeroineEnabled(box, enabled);
  await fillHeroineFolders(box);
  setHeroineEnabled(box, enabled);
  if (enabled && box.querySelector('[data-role="heroine-folder"]').value !== CUSTOM_FOLDER) loadHeroineImages(box);
}

async function loadHeroineImages(box) {
  const dir = box.querySelector('[name="heroine_dir"]').value.trim();
  const slot = box.querySelector(".heroine-images");
  const selected = slot.dataset.selected || slot.querySelector("input:checked")?.value || "";
  slot.innerHTML = `<span class="meta">불러오는 중…</span>`;
  try {
    const res = await api(`/api/heroine/images?dir=${encodeURIComponent(dir)}`);
    box.querySelector('[name="heroine_dir"]').value = res.dir;
    // 직접 입력한 경로가 목록에 있는 폴더면 드롭다운도 그 폴더로
    const select = box.querySelector('[data-role="heroine-folder"]');
    if ([...select.options].some((o) => o.value === res.dir)) {
      select.value = res.dir;
      box.querySelector(".heroine-custom").hidden = true;
    }
    const pick = res.images.some((i) => i.name === selected) ? selected : res.images[0].name;
    slot.dataset.selected = "";  // 처음 한 번만 저장된 대표 이미지를 고르고, 이후엔 사용자가 고른 값 유지
    slot.innerHTML = res.images.map((img) => `
      <label class="heroine-thumb" title="${escapeHtml(img.name)}">
        <input type="radio" name="heroine_ref" value="${escapeHtml(img.name)}" ${img.name === pick ? "checked" : ""}>
        <img src="${img.url}" alt=""></label>`).join("");
  } catch (err) {
    slot.innerHTML = `<span class="warn">${escapeHtml(err.message)}</span>`;
  }
}

function heroinePayload(box) {
  return {
    heroine_enabled: box.querySelector('[name="heroine_enabled"]').checked,
    heroine_dir: box.querySelector('[name="heroine_dir"]').value.trim(),
    heroine_ref: box.querySelector('[name="heroine_ref"]:checked')?.value || "",
  };
}

// 체크하면 폴더 드롭다운이 활성화되고 이미지 목록을 불러온다 (생성 폼·에피소드 설정 공통)
document.addEventListener("change", (e) => {
  const toggle = e.target.closest?.('[data-role="heroine-toggle"]');
  if (toggle) {
    const box = toggle.closest("[data-heroine]");
    setHeroineEnabled(box, toggle.checked);
    if (toggle.checked && !box.querySelector(".heroine-images input")
        && box.querySelector('[data-role="heroine-folder"]').value !== CUSTOM_FOLDER) loadHeroineImages(box);
    return;
  }
  const select = e.target.closest?.('[data-role="heroine-folder"]');
  if (select) {
    const box = select.closest("[data-heroine]");
    const custom = select.value === CUSTOM_FOLDER;
    box.querySelector(".heroine-custom").hidden = !custom;
    box.querySelector(".heroine-images").innerHTML = "";
    if (custom) {
      box.querySelector('[name="heroine_dir"]').focus();
    } else {
      box.querySelector('[name="heroine_dir"]').value = select.value;
      loadHeroineImages(box);
    }
  }
});
document.addEventListener("click", (e) => {
  const btn = e.target.closest?.('[data-action="load-heroine"]');
  if (btn) loadHeroineImages(btn.closest("[data-heroine]"));
});

function renderScene(s, busy, ep) {
  return `
    <li data-script-id="${s.id}">
      <div class="scene-view">
        <div class="row">
          <div>
            <span class="badge section-${s.section}">${s.section.toUpperCase()}</span>
            <b>[${s.duration_sec}s] ${escapeHtml(s.on_screen_text)}</b>
            ${poseBadge(s)}
            ${ep?.heroine_enabled && s.heroine ? `<span class="badge heroine">여자주인공</span>` : ""}
            <span class="scene-emoji" title="${s.emoji ? "장면 스티커 이모지" : "비어 있으면 렌더링 시 내용에 맞게 자동 선택"}">${s.emoji ? escapeHtml(s.emoji) : `<span class="meta">스티커 자동</span>`}</span>
          </div>
          <button class="secondary small" data-action="edit-scene" data-id="${s.id}" ${busy ? "disabled" : ""}>수정</button>
        </div>
        <div>${escapeHtml(s.narration)}</div>
        <div class="scene-media">
          ${s.image_path
            ? `<img class="thumb" src="${s.image_path}" alt="배경 이미지">`
            : `<div class="thumb empty" title="이미지가 없으면 섹션 색 배경으로 렌더링됩니다">기본 배경</div>`}
          <div>
            <div class="meta prompt">영상 프롬프트: ${s.visual_prompt ? escapeHtml(s.visual_prompt) : "(비어 있음 — 섹션별 기본 장면)"}</div>
            <div class="scene-audio">
              <label class="button secondary small ${busy ? "disabled" : ""}">배경 이미지
                <input type="file" accept="image/*" data-action="upload-image" data-id="${s.id}" hidden ${busy ? "disabled" : ""}></label>
              ${s.image_path ? `<button class="secondary small" data-action="delete-image" data-id="${s.id}" ${busy ? "disabled" : ""}>이미지 제거</button>` : ""}
            </div>
          </div>
        </div>
      </div>
    </li>`;
}

function sceneEditorHtml(s, ep) {
  return `
    <form class="scene-editor" data-id="${s.id}">
      <span class="badge section-${s.section}">${s.section.toUpperCase()}</span>
      <label>내레이션 <span class="meta">(바꾸면 이 씬의 음성은 지워지고 2단계에서 다시 만듭니다)</span>
        <textarea name="narration" rows="3" required>${escapeHtml(s.narration)}</textarea></label>
      <label>화면 자막 <input name="on_screen_text" value="${escapeHtml(s.on_screen_text)}"></label>
      <label>스티커 이모지 (1~2개, 비우면 자동) <input name="emoji" maxlength="16" value="${escapeHtml(s.emoji)}"></label>
      ${poseSelect(s)}
      ${ep?.heroine_enabled ? `<label class="inline"><input type="checkbox" name="heroine" ${s.heroine ? "checked" : ""}> 이 씬에 여자주인공 등장</label>` : ""}
      <label>영상 프롬프트 <span class="meta">(Luma 장면 묘사 — 장소·구도·조명·카메라, 영어 권장. 비우면 섹션별 기본 장면)</span>
        <textarea name="visual_prompt" rows="3" placeholder="예: Vertical 9:16 cinematography, rainy neon alley at night, slow push-in, moody blue light">${escapeHtml(s.visual_prompt)}</textarea></label>
      <label>길이(초) <input name="duration_sec" type="number" step="0.1" min="0.5" max="60" value="${s.duration_sec}"></label>
      <div class="actions">
        <button type="submit">저장</button>
        <button type="button" class="secondary" data-action="cancel-edit">취소</button>
      </div>
    </form>`;
}

// ---------- 2단계: 음성(TTS) 제작 ----------
function voiceStep(ep, busy) {
  const total = ep.scripts.length;
  const voiced = ep.scripts.filter((s) => s.audio_path);
  const audioTotal = voiced.reduce((sum, s) => sum + (s.audio_duration_sec || 0), 0);
  const voicing = ep.status === "voicing";
  const pct = total ? Math.round((voiced.length / total) * 100) : 0;
  const rows = ep.scripts.map((s, i) => `
    <li data-audio-id="${s.id}">
      <span class="badge section-${s.section}">${i + 1}</span>
      <span class="voice-text" title="${escapeHtml(s.narration)}">${escapeHtml(s.narration)}</span>
      <span class="voice-player">
        ${s.audio_path
          ? `<audio src="${s.audio_path}" controls preload="none"></audio>
             ${providerBadge(s.audio_provider)}<span class="meta">${s.audio_duration_sec ?? "?"}s${s.audio_duration_sec > s.duration_sec ? ` <span class="warn">씬 길이(${s.duration_sec}s) 초과</span>` : ""}</span>`
          : `<span class="meta">${voicing ? "대기 중…" : "음성 없음"}</span>`}
        <button class="secondary small" data-action="scene-tts" data-id="${s.id}" ${busy ? "disabled" : ""}>
          ${s.audio_path ? "다시 생성" : "생성"}</button>
      </span>
    </li>`).join("");
  return `
    <section class="step ${voicing ? "active" : ""}">
      <div class="stage-title">2단계 · 음성(TTS) 제작
        <span class="meta">확정된 대본으로 씬별 음성을 만들고 미리 듣기</span></div>
      <div class="progress" title="${voiced.length}/${total}개"><div style="width:${pct}%"></div></div>
      <div class="meta">${voicing ? "생성 중 — " : ""}음성 ${voiced.length}/${total}개 · 합계 ${audioTotal.toFixed(1)}s / 목표 ${ep.duration_sec}s
        ${audioTotal > ep.duration_sec ? `<span class="warn">(목표 초과)</span>` : ""}</div>
      <ol class="voice-list">${rows}</ol>
      <div class="stage-actions">
        <label class="inline"><input type="checkbox" data-role="tts-force" ${busy ? "disabled" : ""}> 모두 다시 생성</label>
        <button data-action="generate-tts" data-id="${ep.id}" ${total && !busy ? "" : "disabled"}>
          ${voicing ? "생성 중…" : "2단계: 음성(TTS) 제작하기"}</button>
        <button data-action="play-all" data-id="${ep.id}" class="secondary" ${voiced.length ? "" : "disabled"}>전체 듣기</button>
      </div>
    </section>`;
}

// ---------- 3단계: 씬 영상 ----------
const CLIP_STATUS = {
  none: ["씬 영상 없음", "clip-none"],
  queued: ["대기 중", "clip-queued"],
  generating: ["생성 중…", "clip-generating"],
  completed: ["준비됨", "clip-completed"],
  failed: ["실패", "clip-failed"],
};
const CLIP_SOURCE = { luma: "Luma", fallback: "대체 클립" };

function clipLine(s) {
  const [label, cls] = CLIP_STATUS[s.clip_status] || [s.clip_status, ""];
  const source = s.clip_path && s.clip_status !== "generating" && s.clip_status !== "queued"
    ? ` · ${CLIP_SOURCE[s.clip_source] || s.clip_source}` : "";
  const stale = s.clip_path && s.clip_current === false
    ? ` <span class="warn" title="클립을 만든 뒤 대본·영상 프롬프트·포즈 또는 생성 방식이 바뀌었습니다. 3단계를 다시 실행하면 새로 만듭니다.">· 다시 생성 필요</span>` : "";
  const error = s.clip_status === "failed" && s.clip_error ? ` <span class="warn">${escapeHtml(s.clip_error)}</span>` : "";
  const link = s.clip_path ? ` <a href="${s.clip_path}" target="_blank">열기</a>` : "";
  const frame = s.clip_frame ? ` <a href="${s.clip_frame}" target="_blank" title="영상 프롬프트로 만든 장면 키프레임">키프레임</a>` : "";
  return `<div class="meta"><span class="badge ${cls}">${label}${source}</span>${stale}${link}${frame}${error}</div>`;
}

// Luma에 실제로 보낼 프롬프트 (사용자 '영상 프롬프트'가 맨 앞, 나머지는 자동으로 덧붙는 연기·카메라·일관성 지시)
function lumaPromptDetails(s) {
  const p = s.luma_prompt;
  if (!p) return "";
  const how = p.mode === "generate"
    ? "① 키프레임: 영상 프롬프트 + JB 포즈 이미지(캐릭터 레퍼런스) → ② 그 키프레임에서 영상 생성"
    : "JB 포즈 이미지를 첫 프레임으로 영상 생성";
  return `<details class="luma-prompt"><summary>Luma 프롬프트 보기 ${p.user_prompt ? "" : '<span class="warn">(영상 프롬프트 비어 있음 → 기본 장면)</span>'}</summary>
    <div class="meta">${how}</div>
    ${p.heroine_refs?.length ? `<div class="meta">여자주인공 레퍼런스: ${p.heroine_refs.map(escapeHtml).join(", ")} (레퍼런스 이미지 2번부터)</div>` : ""}
    ${p.heroine_unsupported ? `<div class="warn">이 API 키로는 여자주인공을 함께 넣을 수 없습니다 (Luma Agents API 키 필요).</div>` : ""}
    ${p.frame ? `<div><b>키프레임</b><pre>${escapeHtml(p.frame)}</pre></div>` : ""}
    <div><b>영상</b><pre>${escapeHtml(p.video)}</pre></div></details>`;
}

function clipSummary(ep) {
  const total = ep.scripts.length;
  const ready = ep.scripts.filter((s) => s.clip_path && s.clip_current !== false
    && !["queued", "generating"].includes(s.clip_status));
  const luma = ready.filter((s) => s.clip_source === "luma").length;
  const working = ep.scripts.filter((s) => ["queued", "generating"].includes(s.clip_status)).length;
  const failed = ep.scripts.filter((s) => s.clip_status === "failed").length;
  return { total, ready: ready.length, luma, fallback: ready.length - luma, working, failed };
}

function clipStep(ep, busy) {
  const c = clipSummary(ep);
  const ai = appStatus?.video_ai;
  const pct = c.total ? Math.round((c.ready / c.total) * 100) : 0;
  const clipping = ep.status === "clipping";
  const lumaNote = ai?.enabled
    ? `Luma ${escapeHtml(ai.model)} · 씬당 ${escapeHtml(ai.duration)} · 실패·크레딧 부족 시 대체 클립`
    : `Luma 꺼짐${ai?.reason ? ` (${escapeHtml(ai.reason)})` : ""} → 대체 클립으로 생성`;
  const rows = ep.scripts.map((s, i) => `
    <li>
      ${s.clip_path
        ? `<video class="thumb clip-thumb" src="${s.clip_path}" muted loop playsinline preload="metadata"
             title="씬 영상 (마우스를 올리면 재생)"></video>`
        : `<div class="thumb empty">${i + 1}</div>`}
      <div><b>${i + 1}. ${escapeHtml(s.on_screen_text)}</b>${clipLine(s)}${lumaPromptDetails(s)}</div>
    </li>`).join("");
  return `
    <section class="step ${clipping ? "active" : ""}">
      <div class="stage-title">3단계 · Luma 씬 영상 <span class="meta">${lumaNote}</span></div>
      <div class="progress" title="${c.ready}/${c.total}개 준비"><div style="width:${pct}%"></div></div>
      <div class="meta">${clipping ? `생성 중 — ${c.ready}/${c.total}개 준비 (진행 ${c.working}개)` : `${c.ready}/${c.total}개 준비`}
        ${c.ready ? ` · Luma ${c.luma} · 대체 ${c.fallback}` : ""}${c.failed ? ` · <span class="warn">실패 ${c.failed}</span>` : ""}</div>
      <ol class="clip-list">${rows}</ol>
      <div class="stage-actions">
        <select data-role="clip-mode" ${busy ? "disabled" : ""} title="생성 방식">
          <option value="auto">Luma (실패 시 대체 클립)</option>
          <option value="luma">Luma만</option>
          <option value="fallback">대체 클립만 (무료)</option>
        </select>
        <label class="inline"><input type="checkbox" data-role="clip-force" ${busy ? "disabled" : ""}> 모두 다시 생성</label>
        <button data-action="generate-clips" data-id="${ep.id}" ${ep.scripts.length && !busy ? "" : "disabled"}>
          ${clipping ? "생성 중…" : "3단계: Luma 씬 영상 생성하기"}</button>
      </div>
    </section>`;
}

// ---------- 4단계: 최종 렌더링 ----------
function renderStep(ep, busy) {
  const rendering = ep.status === "rendering";
  const stale = ep.video_path && ep.status !== "done" && !rendering
    ? `<div class="meta warn">최종 영상을 만든 뒤 대본·음성·씬 영상이 바뀌었습니다. 다시 렌더링하면 반영됩니다.</div>` : "";
  const missingVoice = ep.scripts.filter((s) => !s.audio_path).length;
  return `
    <section class="step ${rendering ? "active" : ""}">
      <div class="stage-title">4단계 · 최종 FFmpeg 렌더링 <span class="meta">씬 영상 + 음성 + BGM 덕킹 + 자막·제목 바</span></div>
      ${rendering ? `<div class="progress indeterminate"><div></div></div><div class="meta">렌더링 중…</div>` : ""}
      ${missingVoice && !rendering ? `<div class="meta warn">음성이 없는 씬 ${missingVoice}개 — 렌더링 전에 자동으로 만들지만, 2단계에서 먼저 듣고 확인하는 것을 권장합니다.</div>` : ""}
      ${stale}
      <div class="stage-actions">
        <select data-role="missing-clip" ${busy ? "disabled" : ""} title="씬 영상이 없는 씬">
          <option value="fallback">빈 씬: 대체 클립</option>
          <option value="card">빈 씬: 캐릭터 카드 연출</option>
        </select>
        <label class="inline" title="대본·영상 프롬프트를 바꾸기 전에 만든 씬 영상도 사용 (해제하면 빈 씬과 같이 처리)">
          <input type="checkbox" data-role="use-outdated" checked ${busy ? "disabled" : ""}> 이전 씬 영상도 사용</label>
        <button data-action="render-final" data-id="${ep.id}" ${ep.scripts.length && !busy ? "" : "disabled"}>
          ${rendering ? "렌더링 중…" : "4단계: 최종 FFmpeg 렌더링 실행"}</button>
      </div>
      ${ep.video_path ? `
        <div class="video-box">
          <video src="${ep.video_path}" controls preload="metadata" playsinline></video>
          <a class="button secondary" href="/api/episodes/${ep.id}/video" download>영상 다운로드 (.mp4)</a>
        </div>` : ""}
    </section>`;
}

function renderEpisode(ep) {
  const busy = BUSY.includes(ep.status);
  const hasScript = ep.scripts.length > 0;
  return `
    <div class="episode" data-episode-id="${ep.id}">
      <div class="row">
        <strong>#${ep.id} ${escapeHtml(ep.title)}</strong>
        <span class="badge ${ep.status}">${STATUS_LABEL[ep.status] || ep.status}</span>
      </div>
      <div class="meta">${escapeHtml(ep.product_name)} · ${ep.duration_sec}초 · ${new Date(ep.created_at).toLocaleString()}</div>
      ${ep.error_message ? `<div class="error">${escapeHtml(ep.error_message)}</div>` : ""}
      ${stepNav(ep)}
      ${scriptStep(ep, busy)}
      ${hasScript ? voiceStep(ep, busy) + clipStep(ep, busy) + renderStep(ep, busy) : ""}
      <div class="actions">
        <button data-action="delete" data-id="${ep.id}" class="danger">에피소드 삭제</button>
      </div>
    </div>`;
}

// ---------- 오디오 재생 ----------
function isMediaPlaying() {
  return Boolean(playAll) ||
    [...$episodes.querySelectorAll("audio, video")].some((m) => !m.paused && !m.ended);
}

// 한 번에 하나만 재생 (play 이벤트는 버블링되지 않으므로 캡처 단계에서 처리)
document.addEventListener("play", (e) => {
  document.querySelectorAll("audio, video").forEach((m) => {
    if (m !== e.target && !m.paused) m.pause();
  });
  // 사용자가 다른 플레이어를 직접 재생하면 전체 듣기 중단
  if (playAll && e.target !== playAll.current) stopPlayAll();
}, true);

let playAll = null; // { items: [audio...], index, current, button }

function startPlayAll(episodeEl, button) {
  stopPlayAll();
  const items = [...episodeEl.querySelectorAll(".voice-list audio")];
  if (!items.length) return;
  playAll = { items, index: -1, current: null, button };
  button.textContent = "정지";
  playNext();
}

function playNext() {
  if (!playAll) return;
  playAll.current?.closest("li")?.classList.remove("playing");
  playAll.index += 1;
  const audio = playAll.items[playAll.index];
  if (!audio) return stopPlayAll();
  playAll.current = audio;
  const li = audio.closest("li");
  li.classList.add("playing");
  li.scrollIntoView({ block: "nearest", behavior: "smooth" });
  audio.currentTime = 0;
  audio.addEventListener("ended", playNext, { once: true });
  audio.play().catch(() => stopPlayAll());
}

function stopPlayAll() {
  if (!playAll) return;
  const { current, button } = playAll;
  playAll = null; // play 이벤트 핸들러보다 먼저 비워 재귀 방지
  if (current) {
    current.removeEventListener("ended", playNext);
    current.pause();
    current.closest("li")?.classList.remove("playing");
  }
  if (button?.isConnected) button.textContent = "전체 듣기";
}

let lastSnapshot = "";

async function loadEpisodes({ force = false } = {}) {
  // 편집 중에는 폴링으로 다시 그리지 않음 (입력 내용 보존)
  if (!force && $episodes.querySelector(".scene-editor, .heroine-edit:not([hidden])")) return;
  // 오디오/영상 재생 중에도 다시 그리지 않음 (재생 끊김 방지)
  if (!force && isMediaPlaying()) return;
  stopPlayAll();
  const episodes = await api("/api/episodes");
  const snapshot = JSON.stringify(episodes);
  if (!force && snapshot === lastSnapshot) return; // 변경 없으면 다시 그리지 않음 (재생 중인 영상 유지)
  lastSnapshot = snapshot;
  scriptsById.clear();
  episodeOfScript.clear();
  episodes.forEach((ep) => ep.scripts.forEach((s) => {
    scriptsById.set(String(s.id), s);
    episodeOfScript.set(String(s.id), ep);
  }));
  $episodes.innerHTML = episodes.length
    ? episodes.map(renderEpisode).join("")
    : `<p class="meta">아직 에피소드가 없습니다.</p>`;
}

$form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = Object.fromEntries(new FormData($form));
  data.duration_sec = Number(data.duration_sec);
  Object.assign(data, heroinePayload($form));
  try {
    await api("/api/episodes", { method: "POST", body: JSON.stringify(data) });
    $form.reset();
    $form.querySelector(".heroine-images").innerHTML = "";
    initHeroineBox($form.querySelector("[data-heroine]"));
    await loadEpisodes();
  } catch (err) {
    alert(err.message);
  }
});

// 백그라운드 단계 실행 (1~4단계 버튼) — 요청 본문은 버튼이 속한 단계 패널의 옵션에서 만든다
const STEP_ACTIONS = {
  script: (id) => ["/api/generate-script", { episode_id: id }],
  "generate-tts": (id, step) => ["/api/generate-tts", {
    episode_id: id, force: step.querySelector('[data-role="tts-force"]').checked }],
  "generate-clips": (id, step) => ["/api/generate-luma-clips", {
    episode_id: id,
    mode: step.querySelector('[data-role="clip-mode"]').value,
    force: step.querySelector('[data-role="clip-force"]').checked }],
  "render-final": (id, step) => ["/api/render-final-video", {
    episode_id: id,
    missing_clip: step.querySelector('[data-role="missing-clip"]').value,
    use_outdated: step.querySelector('[data-role="use-outdated"]').checked }],
};

function confirmStep(action, body, episodeEl) {
  if (action === "script" && episodeEl.querySelector(".scenes li")) {
    return confirm("대본을 다시 만들면 지금 씬들과 씬별 음성이 새 대본으로 바뀝니다. 진행할까요?");
  }
  if (action === "generate-clips" && body.mode !== "fallback" && appStatus?.video_ai?.enabled) {
    return confirm("Luma 크레딧이 사용됩니다 (이미 최신 씬 영상이 있는 씬은 건너뜀). 진행할까요?");
  }
  return true;
}

$episodes.addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-action]");
  if (!btn) return;
  const { action, id } = btn.dataset;

  if (action === "edit-scene") {
    const li = btn.closest("li");
    li.querySelector(".scene-view").hidden = true;
    li.insertAdjacentHTML("beforeend", sceneEditorHtml(scriptsById.get(id), episodeOfScript.get(id)));
    return;
  }
  if (action === "cancel-edit") {
    const li = btn.closest("li");
    li.querySelector(".scene-editor").remove();
    li.querySelector(".scene-view").hidden = false;
    if (!$episodes.querySelector(".scene-editor")) loadEpisodes(); // 편집 중 밀린 갱신 반영
    return;
  }
  if (action === "edit-heroine" || action === "cancel-heroine") {
    const step = btn.closest(".step");
    const form = step.querySelector(".heroine-edit");
    form.hidden = action === "cancel-heroine";
    step.querySelector(".heroine-summary").hidden = !form.hidden;
    if (!form.hidden && !form.dataset.foldersLoaded) initHeroineBox(form);
    if (form.hidden && !$episodes.querySelector(".scene-editor, .heroine-edit:not([hidden])")) loadEpisodes();
    return;
  }
  if (action === "play-all") {
    if (playAll?.button === btn) stopPlayAll();
    else startPlayAll(btn.closest(".episode"), btn);
    return;
  }

  if (action === "delete-image") {
    try {
      await api(`/api/scripts/${id}/image`, { method: "DELETE" });
      await loadEpisodes({ force: true });
    } catch (err) {
      alert(err.message);
    }
    return;
  }

  // 2단계: 씬 하나 음성 생성 (즉시) → 생성되면 바로 재생
  if (action === "scene-tts") {
    btn.disabled = true;
    btn.textContent = "생성 중…";
    try {
      await api(`/api/scripts/${id}/tts`, { method: "POST" });
      await loadEpisodes({ force: true });
      document.querySelector(`li[data-audio-id="${id}"] audio`)?.play();
    } catch (err) {
      alert(err.message);
      btn.disabled = false;
      btn.textContent = "생성";
    }
    return;
  }

  if (STEP_ACTIONS[action]) {
    const episodeEl = btn.closest(".episode");
    const [path, body] = STEP_ACTIONS[action](Number(id), btn.closest(".step"));
    if (!confirmStep(action, body, episodeEl)) return;
    btn.disabled = true;
    try {
      const res = await api(path, { method: "POST", body: JSON.stringify(body) });
      showToast(res.message);
      await loadEpisodes({ force: true });
    } catch (err) {
      alert(err.message);
      btn.disabled = false;
    }
    return;
  }

  if (action === "delete") {
    if (!confirm("에피소드를 삭제할까요? 음성·씬 영상·최종 영상 파일도 함께 지워집니다.")) return;
    try {
      await api(`/api/episodes/${id}`, { method: "DELETE" });
      await loadEpisodes({ force: true });
    } catch (err) {
      alert(err.message);
    }
  }
});

// 씬 배경 이미지 업로드
$episodes.addEventListener("change", async (e) => {
  const input = e.target.closest('input[data-action="upload-image"]');
  if (!input || !input.files.length) return;
  const body = new FormData();
  body.append("file", input.files[0]);
  input.closest("label").classList.add("disabled");
  try {
    const res = await fetch(`/api/scripts/${input.dataset.id}/image`, { method: "POST", body });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
    await loadEpisodes({ force: true }); // 상태(재렌더링 필요) 갱신
  } catch (err) {
    alert(err.message);
    input.closest("label")?.classList.remove("disabled");
  }
});

// 1단계: 씬 대본 저장 (음성은 만들지 않음 — 내레이션이 바뀌면 그 씬 음성은 지워지고 2단계에서 다시 생성)
$episodes.addEventListener("submit", async (e) => {
  const heroineForm = e.target.closest(".heroine-edit");
  if (heroineForm) {
    e.preventDefault();
    try {
      await api(`/api/episodes/${heroineForm.dataset.episodeId}`, { method: "PATCH",
        body: JSON.stringify(heroinePayload(heroineForm)) });
      heroineForm.hidden = true;
      await loadEpisodes({ force: true });
    } catch (err) {
      alert(err.message);
    }
    return;
  }
  const form = e.target.closest(".scene-editor");
  if (!form) return;
  e.preventDefault();
  const data = Object.fromEntries(new FormData(form));
  data.duration_sec = Number(data.duration_sec);
  const heroineBox = form.querySelector('[name="heroine"]');
  if (heroineBox) data.heroine = heroineBox.checked; else delete data.heroine;
  const buttons = form.querySelectorAll("button");
  buttons.forEach((b) => (b.disabled = true));
  try {
    await api(`/api/scripts/${form.dataset.id}`, { method: "PATCH", body: JSON.stringify(data) });
    const li = form.closest("li");
    form.remove(); // 편집기를 닫아야 목록을 다시 그릴 수 있음
    li.querySelector(".scene-view").hidden = false;
    await loadEpisodes({ force: !$episodes.querySelector(".scene-editor") });
  } catch (err) {
    alert(err.message);
    buttons.forEach((b) => (b.disabled = false));
  }
});

document.getElementById("refresh").addEventListener("click", () => loadEpisodes({ force: true }));
initHeroineBox($form.querySelector("[data-heroine]"));  // 생성 폼의 캐릭터 폴더 목록 (체크 전에는 비활성)
loadStatus().then(() => loadEpisodes());
setInterval(() => { loadStatus(); loadEpisodes(); }, 5000);

// 작업 시작 알림 (alert 대신 잠깐 떠 있다 사라지는 안내)
function showToast(message) {
  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = message;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4500);
}

// 씬 영상 썸네일: 마우스를 올리면 재생
document.addEventListener("mouseover", (e) => {
  const v = e.target.closest?.("video.clip-thumb");
  if (v) v.play().catch(() => {});
});
document.addEventListener("mouseout", (e) => {
  const v = e.target.closest?.("video.clip-thumb");
  if (v) v.pause();
});
