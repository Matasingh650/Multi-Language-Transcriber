// const API_BASE = "http://127.0.0.1:8000";

// const form = document.getElementById("transcribeForm");
// const dropzone = document.getElementById("dropzone");
// const fileInput = document.getElementById("fileInput");
// const browseBtn = document.getElementById("browseBtn");
// const progress = document.getElementById("progress");
// const bar = document.getElementById("bar");
// const statusEl = document.getElementById("status");
// const resultEl = document.getElementById("result");
// const langEl = document.getElementById("lang");
// const durEl = document.getElementById("dur");
// const previewEl = document.getElementById("preview");
// const downloadsEl = document.getElementById("downloads");
// const translateSelect = document.getElementById("translateTo");

// async function loadLanguages(){
//   const r = await fetch(`${API_BASE}/languages`);
//   const data = await r.json();
//   for(const code of data.codes){
//     const opt = document.createElement("option");
//     opt.value = code; opt.textContent = code;
//     translateSelect.appendChild(opt);
//   }
// }
// loadLanguages();

// browseBtn.addEventListener("click", () => fileInput.click());

// ["dragenter","dragover"].forEach(evt => {
//   dropzone.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dropzone.classList.add("drag"); });
// });
// ["dragleave","drop"].forEach(evt => {
//   dropzone.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dropzone.classList.remove("drag"); });
// });
// dropzone.addEventListener("click", () => fileInput.click());
// dropzone.addEventListener("drop", e => {
//   const file = e.dataTransfer.files?.[0];
//   if (file) fileInput.files = e.dataTransfer.files;
// });

// function setProgress(pct, label){
//   progress.classList.remove("hidden");
//   bar.style.width = `${pct}%`;
//   statusEl.textContent = label || "";
// }

// form.addEventListener("submit", async (e) => {
//   e.preventDefault();
//   if (!fileInput.files.length) { alert("Please choose a file."); return; }

//   resultEl.classList.add("hidden");
//   downloadsEl.innerHTML = "";
//   setProgress(5, "Uploading…");

//   const fd = new FormData(form);
//   fd.set("make_txt", form.make_txt.checked);
//   fd.set("make_srt", form.make_srt.checked);
//   fd.set("make_vtt", form.make_vtt.checked);
//   fd.set("make_json", form.make_json.checked);

//   // Manual upload (shows progress)
//   const xhr = new XMLHttpRequest();
//   xhr.open("POST", `${API_BASE}/transcribe`, true);
//   xhr.upload.onprogress = (e) => {
//     if (e.lengthComputable) {
//       const pct = Math.min(90, Math.round((e.loaded / e.total) * 90));
//       setProgress(pct, "Uploading…");
//     }
//   };
//   xhr.onreadystatechange = () => {
//     if (xhr.readyState === 4) {
//       if (xhr.status >= 200 && xhr.status < 300) {
//         setProgress(100, "Done");
//         const data = JSON.parse(xhr.responseText);
//         renderResult(data);
//       } else {
//         alert("Error: " + xhr.responseText);
//       }
//     }
//   };
//   xhr.send(fd);
// });

// function renderResult(data){
//   resultEl.classList.remove("hidden");
//   langEl.textContent = data.detected_language || "unknown";
//   durEl.textContent = Math.round(data.duration_sec ?? 0);
//   previewEl.textContent = data.text_preview || "";

//   downloadsEl.innerHTML = "";

//   const makeLinks = (variantName, obj) => {
//     const title = document.createElement("h3");
//     title.textContent = variantName === "original" ? "Original language" : `Translated (${variantName.split("_").pop()})`;
//     downloadsEl.appendChild(title);

//     const keys = Object.keys(obj);
//     if (!keys.length) {
//       const p = document.createElement("p"); p.textContent = "No files generated.";
//       downloadsEl.appendChild(p);
//       return;
//     }
//     for (const k of keys) {
//       const a = document.createElement("a");
//       a.href = `${API_BASE}${obj[k]}`;
//       a.download = ""; // let browser download
//       a.textContent = k.toUpperCase();
//       downloadsEl.appendChild(a);
//     }
//   };

//   if (data.downloads?.original) makeLinks("original", data.downloads.original);
//   if (data.downloads?.translated) {
//     // translated keys like stem_en.txt etc.; we just print as provided
//     makeLinks("translated", data.downloads.translated);
//   }
// }


// frontend/app.js
const API_BASE = "http://127.0.0.1:8000";

const form = document.getElementById("transcribeForm");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const progress = document.getElementById("progress");
const bar = document.getElementById("bar");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const langEl = document.getElementById("lang");
const durEl = document.getElementById("dur");
const previewEl = document.getElementById("preview");
const downloadsEl = document.getElementById("downloads");
const translateSelect = document.getElementById("translateTo");

// Show selected filename in the dropzone
const fileNameEl = document.createElement("div");
fileNameEl.style.marginTop = "8px";
fileNameEl.style.color = "#cbd5e1";
dropzone.appendChild(fileNameEl);

function showChosenFile() {
  if (fileInput.files && fileInput.files.length) {
    fileNameEl.innerHTML = `Selected: <strong>${fileInput.files[0].name}</strong>`;
  } else {
    fileNameEl.textContent = "";
  }
}

// Load target languages
async function loadLanguages(){
  try {
    const r = await fetch(`${API_BASE}/languages`);
    if (!r.ok) return;
    const data = await r.json();
    for (const code of data.codes) {
      const opt = document.createElement("option");
      opt.value = code; opt.textContent = code;
      translateSelect.appendChild(opt);
    }
  } catch (_) { /* ignore if backend not ready */ }
}
loadLanguages();

browseBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", showChosenFile);

// Drag & drop wiring
["dragenter","dragover"].forEach(evt => {
  dropzone.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dropzone.classList.add("drag"); });
});
["dragleave","drop"].forEach(evt => {
  dropzone.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dropzone.classList.remove("drag"); });
});
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("drop", e => {
  const list = e.dataTransfer?.files;
  if (list && list.length) {
    fileInput.files = list;
    fileInput.dispatchEvent(new Event("change")); // <- important
  }
});

function setProgress(pct, label){
  progress.classList.remove("hidden");
  bar.style.width = `${pct}%`;
  statusEl.textContent = label || "";
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  if (!fileInput.files.length) { alert("Please choose a file."); return; }

  resultEl.classList.add("hidden");
  downloadsEl.innerHTML = "";
  setProgress(5, "Uploading…");

  const fd = new FormData(form);
  // Normalize checkboxes to booleans
  fd.set("make_txt", form.make_txt.checked);
  fd.set("make_srt", form.make_srt.checked);
  fd.set("make_vtt", form.make_vtt.checked);
  fd.set("make_json", form.make_json.checked);

  const xhr = new XMLHttpRequest();
  xhr.open("POST", `${API_BASE}/transcribe`, true);

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const pct = Math.min(90, Math.round((e.loaded / e.total) * 90));
      setProgress(pct, "Uploading…");
    }
  };
  xhr.onerror = () => alert("Network error while uploading.");

  xhr.onreadystatechange = () => {
    if (xhr.readyState === 4) {
      if (xhr.status >= 200 && xhr.status < 300) {
        setProgress(100, "Done");
        const data = JSON.parse(xhr.responseText);
        renderResult(data);
      } else {
        // Try to show API error detail if present
        try {
          const err = JSON.parse(xhr.responseText);
          alert("Error: " + (err.detail || xhr.responseText));
        } catch {
          alert("Error: " + xhr.responseText);
        }
      }
    }
  };

  xhr.send(fd);
});

function renderResult(data){
  resultEl.classList.remove("hidden");
  langEl.textContent = data.detected_language || "unknown";
  durEl.textContent = Math.round(data.duration_sec ?? 0);
  previewEl.textContent = data.text_preview || "";
  downloadsEl.innerHTML = "";

  const makeLinks = (titleText, filesObj) => {
    const title = document.createElement("h3");
    title.textContent = titleText;
    downloadsEl.appendChild(title);

    const keys = Object.keys(filesObj || {});
    if (!keys.length) {
      const p = document.createElement("p"); p.textContent = "No files generated.";
      downloadsEl.appendChild(p);
      return;
    }
    for (const k of keys) {
      const a = document.createElement("a");
      a.href = `${API_BASE}${filesObj[k]}`;
      a.download = ""; // hint to download
      a.textContent = k.toUpperCase();
      downloadsEl.appendChild(a);
    }
  };

  if (data.downloads?.original) makeLinks("Original language", data.downloads.original);
  if (data.downloads?.translated) makeLinks("Translated", data.downloads.translated);
}
