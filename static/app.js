/* Kleenly front end. Plain JS, no framework.
   Every value that comes from the uploaded file is inserted with textContent, never innerHTML.
   innerHTML is only used for the small fixed icons defined below. */
(() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const fmt = (n) => Number(n).toLocaleString("en-IN");
  const plural = (n, word) => `${fmt(n)} ${word}${n === 1 ? "" : "s"}`;

  // ------------------------------------------------------------------ helpers
  // el("div", {class: "x", text: "hi"}, child, child)
  function el(tag, props = {}, ...kids) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (v == null || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return node;
  }

  const ICONS = {
    table: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18M9 4v16"/>',
    check: '<circle cx="12" cy="12" r="9"/><path d="m8 12.5 2.8 2.8L16 9.5"/>',
    copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/>',
    alert: '<path d="M12 4 3 20h18L12 4Z"/><path d="M12 10v4M12 17.5v.01"/>',
    download: '<path d="M12 4v11m0 0-4-4m4 4 4-4"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
    list: '<path d="M8 6h12M8 12h12M8 18h12M4 6h.01M4 12h.01M4 18h.01"/>',
  };
  function icon(name) {
    const t = document.createElement("template");
    t.innerHTML = `<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name]}</svg>`;
    return t.content.firstChild;
  }

  // ---------------------------------------------------------------- elements
  const upload = $("#upload");
  const dropPrompt = $("#dropPrompt");
  const fileChip = $("#fileChip");
  const fileInput = $("#file");
  const runBtn = $("#run");
  const runHint = $("#runHint");
  const errorBox = $("#error");
  const results = $("#results");
  const customNote = $("#customNote");
  const viewHome = $("#viewHome");
  const viewResults = $("#viewResults");
  function showView(name) {
    viewHome.hidden = name !== "home";
    viewResults.hidden = name !== "results";
    document.body.classList.toggle("on-results", name === "results");
    window.scrollTo({ top: 0, behavior: "auto" });
  }
  let currentFile = null;

  // Replay the header animation.
  $("#replay").addEventListener("click", () => {
    const grid = $("#demoGrid");
    grid.classList.remove("play");
    void grid.offsetWidth; // restart the CSS animation
    grid.classList.add("play");
  });

  // ------------------------------------------------------------- file picking
  function setFile(file) {
    currentFile = file;
    dropPrompt.hidden = !!file;
    fileChip.hidden = !file;
    if (file) {
      $("#fileName").textContent = file.name;
      $("#fileSize").textContent = file.size > 102400 ? `${Math.round(file.size / 1024)} KB` : `${(file.size / 1024).toFixed(1)} KB`;
    }
    runBtn.disabled = !file;
    runHint.hidden = !!file;
    showError("");
  }

  $("#browse").addEventListener("click", () => fileInput.click());
  $("#changeFile").addEventListener("click", () => fileInput.click());
  $("#removeFile").addEventListener("click", () => { fileInput.value = ""; setFile(null); });
  fileInput.addEventListener("change", () => fileInput.files[0] && setFile(fileInput.files[0]));

  ["dragenter", "dragover"].forEach((ev) => upload.addEventListener(ev, (e) => { e.preventDefault(); upload.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => upload.addEventListener(ev, (e) => { e.preventDefault(); upload.classList.remove("over"); }));
  upload.addEventListener("drop", (e) => e.dataTransfer.files[0] && setFile(e.dataTransfer.files[0]));

  $("#sample").addEventListener("click", async () => {
    try {
      const res = await fetch("/api/sample");
      if (!res.ok) throw new Error();
      setFile(new File([await res.blob()], "messy_customers.csv", { type: "text/csv" }));
    } catch {
      showError("Could not load the sample file. Is the server still running?");
    }
  });

  function showError(msg) {
    errorBox.hidden = !msg;
    errorBox.textContent = msg || "";
  }

  // ------------------------------------------------------------------ presets
  const PRESETS = {
    review: { numeric_impute: "median", categorical_impute: "mode", col_missing: 60, row_missing: 50, fuzzy_duplicates: "flag", outlier_action: "flag" },
    autofix: { numeric_impute: "knn", categorical_impute: "mode", col_missing: 60, row_missing: 50, fuzzy_duplicates: "remove", outlier_action: "cap" },
    strict: { numeric_impute: "median", categorical_impute: "unknown", col_missing: 40, row_missing: 30, fuzzy_duplicates: "remove", outlier_action: "remove" },
  };
  const BASE = { outlier_method: "iqr", iqr_k: 1.5, fuzzy_threshold: 95 };

  function applyPreset(name) {
    const p = { ...BASE, ...PRESETS[name] };
    for (const [id, value] of Object.entries(p)) $(`#${id}`).value = value;
    customNote.hidden = true;
  }

  document.querySelectorAll('input[name="preset"]').forEach((r) =>
    r.addEventListener("change", () => r.checked && applyPreset(r.value)));

  // Any manual edit of a detailed rule means "custom".
  $("#options").addEventListener("input", () => {
    document.querySelectorAll('input[name="preset"]').forEach((r) => { r.checked = false; });
    customNote.hidden = false;
  });

  function readOptions() {
    const pct = (id) => Number($(id).value) / 100;
    return {
      numeric_impute: $("#numeric_impute").value,
      categorical_impute: $("#categorical_impute").value,
      col_missing_threshold: pct("#col_missing"),
      row_missing_threshold: pct("#row_missing"),
      drop_exact_duplicates: $("#drop_exact_duplicates").checked,
      fuzzy_duplicates: $("#fuzzy_duplicates").value,
      fuzzy_threshold: Number($("#fuzzy_threshold").value),
      dayfirst: $("#dayfirst").checked,
      standardize_categories: $("#standardize_categories").checked,
      outlier_method: $("#outlier_method").value,
      iqr_k: Number($("#iqr_k").value),
      outlier_action: $("#outlier_action").value,
      isolation_forest: $("#isolation_forest").checked,
    };
  }

  // ---------------------------------------------------------------------- run
  runBtn.addEventListener("click", async () => {
    if (!currentFile) return;
    showError("");
    runBtn.disabled = true;
    $("#runLabel").textContent = "Cleaning...";
    showView("results");
    results.replaceChildren(el("div", { class: "busy" },
      el("div", { class: "spinner" }),
      el("p", { text: "Cleaning your file" }),
      el("span", { text: "Checking missing values, duplicates, types and outliers." })));
    try {
      const body = new FormData();
      body.append("file", currentFile);
      body.append("options", JSON.stringify(readOptions()));
      const res = await fetch("/api/clean", { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || `The server returned an error (${res.status}).`);
      render(data.job_id, data.report);
    } catch (err) {
      results.replaceChildren();
      showView("home");
      showError(err.message || "Something went wrong. Try again.");
    } finally {
      runBtn.disabled = false;
      $("#runLabel").textContent = "Clean my file";
    }
  });

  // ---------------------------------------------------------------- rendering
  function cell(v) {
    return v == null || v === ""
      ? el("span", { class: "empty-cell", text: "empty" })
      : el("span", { class: "val", text: String(v) });
  }

  function sentence(r) {
    const c = Object.fromEntries(r.summary.map((s) => [s.category, s.count]));
    const n = (k) => c[k] || 0;
    const fixed = [];
    const dups = n("duplicate_exact") + n("near_duplicate_removed");
    if (dups) fixed.push(`${plural(dups, "duplicate row")} removed`);
    const sparse = n("row_dropped_sparse") + n("outlier_removed");
    if (sparse) fixed.push(`${plural(sparse, "row")} dropped`);
    if (n("column_dropped")) fixed.push(`${plural(n("column_dropped"), "empty column")} dropped`);
    if (n("missing_filled")) fixed.push(`${plural(n("missing_filled"), "missing value")} filled`);
    if (n("type_converted")) fixed.push(`${plural(n("type_converted"), "column")} converted to the right type`);
    if (n("invalid_value")) fixed.push(`${plural(n("invalid_value"), "invalid value")} set to missing`);
    if (n("category_standardized")) fixed.push(`${plural(n("category_standardized"), "spelling")} unified`);
    if (n("outlier_capped")) fixed.push(`${plural(n("outlier_capped"), "outlier")} capped`);

    const check = [];
    if (n("outlier_flagged")) check.push(`${plural(n("outlier_flagged"), "outlier")}`);
    if (n("outlier_multivariate")) check.push(`${plural(n("outlier_multivariate"), "unusual row")}`);
    if (n("near_duplicate_flagged")) check.push(`${plural(n("near_duplicate_flagged"), "possible near-duplicate")}`);

    const p = el("p", { class: "sentence" });
    if (!fixed.length && !check.length) {
      p.append("No problems found. The file came out the same as it went in.");
      return p;
    }
    if (fixed.length) p.append(el("b", { class: "s-fix", text: "Fixed: " }), `${fixed.join(", ")}. `);
    if (check.length) p.append(el("b", { class: "s-flag", text: "Worth a look: " }), `${check.join(", ")}.`);
    return p;
  }

  // The five checks, in the order a reader thinks about them.
  function steps(r) {
    const c = Object.fromEntries(r.summary.map((s) => [s.category, s.count]));
    const n = (k) => c[k] || 0;
    const say = (parts) => parts.filter(Boolean).join(", ");
    const defs = [
      ["Missing values", say([
        n("fake_null_normalized") && `${plural(n("fake_null_normalized"), "placeholder")} (N/A, -, null) turned into blanks`,
        n("missing_filled") && `${plural(n("missing_filled"), "gap")} filled`,
        n("row_dropped_sparse") && `${plural(n("row_dropped_sparse"), "empty row")} dropped`,
        n("column_dropped") && `${plural(n("column_dropped"), "empty column")} dropped`])],
      ["Data types", say([
        n("type_converted") && `${plural(n("type_converted"), "column")} converted`,
        n("format_standardized") && `${plural(n("format_standardized"), "value")} reformatted`,
        n("invalid_value") && `${plural(n("invalid_value"), "invalid value")} set to blank`])],
      ["Spelling", say([
        n("category_standardized") && `${plural(n("category_standardized"), "spelling")} unified`,
        n("whitespace_trimmed") && `${plural(n("whitespace_trimmed"), "stray space")} trimmed`])],
      ["Duplicates", say([
        (n("duplicate_exact") + n("near_duplicate_removed")) && `${plural(n("duplicate_exact") + n("near_duplicate_removed"), "row")} removed`,
        n("near_duplicate_flagged") && `${plural(n("near_duplicate_flagged"), "possible match")} flagged for you`])],
      ["Outliers", say([
        n("outlier_flagged") && `${plural(n("outlier_flagged"), "value")} flagged`,
        n("outlier_multivariate") && `${plural(n("outlier_multivariate"), "unusual row")} flagged`,
        n("outlier_capped") && `${plural(n("outlier_capped"), "value")} capped`,
        n("outlier_removed") && `${plural(n("outlier_removed"), "row")} removed`])],
    ];
    return el("ol", { class: "flow" }, defs.map(([title, text], i) =>
      el("li", { class: text ? "did" : "none" },
        el("span", { class: "step-no", text: String(i + 1) }),
        el("div", {}, el("strong", { text: title }), el("span", { text: text || "Nothing to fix here." })))));
  }

  function render(jobId, r) {
    const t = r.totals;
    results.replaceChildren(
      el("div", { class: "done" },
        el("h2", { text: "Your CSV has been cleaned!" }),
        el("p", { class: "meta", text: `${plural(t.rows_after, "clean row")} ready from ${r.file.name || "your file"}` }),
        sentence(r),
        el("div", { class: "downloads" },
          el("a", { class: "btn main", href: `/api/download/${jobId}` }, icon("download"), "Download cleaned CSV"),
          el("a", { class: "btn", href: `/api/changes/${jobId}` }, icon("list"), "Change log"),
          el("a", { class: "btn", href: `/api/report/${jobId}` }, "Report (JSON)")),
        el("p", { class: "meta", text: `${fmt(t.total_changes)} changes recorded` })),
      el("h3", { class: "section-title", text: "The five steps we went through" }),
      steps(r),
      tiles(r),
      el("details", { class: "board-wrap" },
        el("summary", { text: "See every change" }),
        tabs(r)),
      el("div", { class: "again" }, el("button", { type: "button", id: "again", class: "btn", text: "Clean another file" })));
    $("#again").addEventListener("click", () => showView("home"));
    requestAnimationFrame(() => requestAnimationFrame(() => {
      results.querySelectorAll(".bar").forEach((b) => b.classList.add("ready"));
    }));
  }

  function tiles(r) {
    const t = r.totals, h = r.headline, c = r.completeness;
    const tile = (cls, iconName, label, value, sub, extra) =>
      el("div", { class: `tile ${cls}` },
        el("dt", {}, icon(iconName), label),
        el("dd", {}, value, sub && el("span", { class: "sub", text: sub }), extra));
    const colsNote = t.cols_after !== t.cols_before ? `${fmt(t.cols_after)} of ${fmt(t.cols_before)} columns kept` : `${fmt(t.cols_before)} columns`;
    return el("dl", { class: "tiles" },
      tile("t-brand", "table", "Rows kept", fmt(t.rows_after), `from ${fmt(t.rows_before)}, ${colsNote}`),
      tile("t-fix", "check", "Complete cells", `${c.after}%`, `was ${c.before}% before cleaning`,
        el("div", { class: "bar", style: `--w:${c.after}%` }, el("i"))),
      tile("t-bad", "copy", "Duplicates", fmt(h.duplicates_removed),
        h.duplicates_flagged ? `removed, ${fmt(h.duplicates_flagged)} more flagged` : "removed"),
      tile("t-flag", "alert", "Outliers", fmt(h.outliers), "found, details below"));
  }

  function tabs(r) {
    const defs = [
      ["Summary", () => summaryPanel(r)],
      ["Changes", () => changesPanel(r)],
      ["Columns", () => columnsPanel(r)],
      ["Preview", () => previewPanel(r)],
    ];
    const bar = el("div", { class: "tabs", role: "tablist" });
    const body = el("div", { class: "panel-body", role: "tabpanel" });
    const buttons = defs.map(([name], i) => {
      const b = el("button", { class: "tab", role: "tab", type: "button", "aria-selected": "false", tabindex: "-1", text: name });
      b.addEventListener("click", () => select(i));
      b.addEventListener("keydown", (e) => {
        if (e.key === "ArrowRight") select((i + 1) % defs.length, true);
        if (e.key === "ArrowLeft") select((i + defs.length - 1) % defs.length, true);
      });
      return b;
    });
    function select(i, focus) {
      buttons.forEach((b, j) => {
        b.setAttribute("aria-selected", String(i === j));
        b.tabIndex = i === j ? 0 : -1;
      });
      body.replaceChildren(defs[i][1]());
      if (focus) buttons[i].focus();
    }
    bar.append(...buttons);
    select(0);
    return el("div", { class: "board" }, bar, body);
  }

  // ------------------------------------------------------------------ summary
  const KIND_TEXT = { fix: "fixed", flag: "flagged", info: "kept" };

  function summaryRow(s, max) {
    return el("li", {},
      el("span", { class: `dot ${s.kind}` }),
      el("span", { text: s.label }),
      el("span", { class: `pill ${s.kind}`, text: KIND_TEXT[s.kind] || "" }),
      el("span", { class: "count", text: fmt(s.count) }),
      el("span", { class: "track" }, el("i", { class: s.kind, style: `width:${Math.max(2, (s.count / max) * 100)}%` })));
  }

  function nearDuplicateTable(pairs) {
    const line = (rowNo, text) =>
      el("div", {}, el("span", { class: "why", text: `Row ${rowNo}: ` }), el("span", { class: "val", text }));
    const row = (p) =>
      el("tr", {},
        el("td", { class: "num", text: `${p.similarity}%` }),
        el("td", {}, line(p.row_a, p.a), line(p.row_b, p.b)));
    const head = el("thead", {}, el("tr", {},
      el("th", { class: "num", text: "Similarity" }),
      el("th", { text: "The two rows" })));
    return el("div", { class: "table-wrap" }, el("table", {}, head, el("tbody", {}, pairs.map(row))));
  }

  function summaryPanel(r) {
    const max = Math.max(1, ...r.summary.map((s) => s.count));
    const wrap = el("div");
    if (!r.summary.length) {
      wrap.append(el("p", { text: "Nothing needed fixing in this file." }));
    } else {
      wrap.append(el("ul", { class: "sum-list" }, r.summary.map((s) => summaryRow(s, max))));
    }
    if (r.totals.flag_column_added) {
      wrap.append(el("p", { class: "note-small", text: 'Flagged rows have a note in the extra "_flags" column of the cleaned file.' }));
    }
    if (r.notes.length) wrap.append(el("div", { class: "notes" }, r.notes.map((n) => el("p", { text: n }))));
    if (r.near_duplicate_pairs.length) {
      wrap.append(el("h3", { class: "subhead", text: "Possible near-duplicates to review" }), nearDuplicateTable(r.near_duplicate_pairs));
    }
    const f = r.file;
    const skipped = f.malformed_lines_skipped ? ` ${f.malformed_lines_skipped} malformed line(s) were skipped.` : "";
    wrap.append(el("p", { class: "file-info", text: `Read as ${f.encoding} text with "${f.delimiter}" as the separator.${skipped}` }));
    return wrap;
  }

  // ------------------------------------------------------------------ changes
  function changeRow(c) {
    return el("tr", {},
      el("td", { class: "num", text: c.row ?? "" }),
      el("td", { text: c.column ?? "" }),
      el("td", {},
        c.old != null ? el("span", { class: "old val", text: c.old }) : null,
        c.old != null && c.new != null ? " " : null,
        c.new != null ? el("span", { class: "new val", text: c.new }) : null,
        c.old == null && c.new == null ? el("span", { class: "empty-cell", text: c.label }) : null),
      el("td", { class: "why", text: c.detail || c.label }));
  }

  function changesPanel(r) {
    const cats = [...new Map(r.changes.map((c) => [c.category, c.label])).entries()];
    const select = el("select", { "aria-label": "Filter by kind of change" },
      el("option", { value: "", text: "All kinds of change" }),
      cats.map(([k, l]) => el("option", { value: k, text: l })));
    const search = el("input", { type: "search", placeholder: "Search rows, columns, values", "aria-label": "Search changes" });
    const tbody = el("tbody");
    const note = el("p", { class: "note-small" });

    function draw() {
      const q = search.value.trim().toLowerCase();
      const rows = r.changes.filter((c) =>
        (!select.value || c.category === select.value) &&
        (!q || [c.row, c.column, c.old, c.new, c.detail, c.label].join(" ").toLowerCase().includes(q)));
      tbody.replaceChildren(...rows.map(changeRow));
      if (!rows.length) tbody.append(el("tr", {}, el("td", { colspan: "4", class: "why", text: "No changes match." })));
      note.textContent = `Showing up to 40 examples per kind of change. The change log CSV has all ${fmt(r.totals.total_changes)}. Row numbers match your spreadsheet, where the header is row 1.`;
    }
    select.addEventListener("change", draw);
    search.addEventListener("input", draw);
    draw();

    const head = el("thead", {}, el("tr", {},
      el("th", { class: "num", text: "Row" }),
      el("th", { text: "Column" }),
      el("th", { text: "Change" }),
      el("th", { text: "Why" })));
    return el("div", {},
      el("div", { class: "tools" }, select, search),
      el("div", { class: "table-wrap" }, el("table", {}, head, tbody)),
      note);
  }

  // ------------------------------------------------------------------ columns
  function columnRow(c) {
    const fixedMissing = !c.dropped && c.missing_after < c.missing_before;
    return el("tr", {},
      el("td", {}, el("span", { class: "val", text: c.name })),
      el("td", {}, c.dropped ? null : el("span", { class: `type ${c.type}`, text: c.type })),
      el("td", { class: "num", text: fmt(c.missing_before) }),
      el("td", { class: `num ${fixedMissing ? "better" : ""}`, text: c.dropped ? "" : fmt(c.missing_after) }),
      el("td", {}, c.dropped
        ? el("span", { class: "dropped", text: "Column dropped (mostly empty)" })
        : el("div", { class: "chips" }, Object.entries(c.changes).map(([k, v]) => el("span", { text: `${k}: ${fmt(v)}` })))));
  }

  function columnsPanel(r) {
    const head = el("thead", {}, el("tr", {},
      el("th", { text: "Column" }),
      el("th", { text: "Type" }),
      el("th", { class: "num", text: "Missing before" }),
      el("th", { class: "num", text: "Missing after" }),
      el("th", { text: "What changed" })));
    return el("div", { class: "table-wrap" }, el("table", {}, head, el("tbody", {}, r.columns.map(columnRow))));
  }

  // ------------------------------------------------------------------ preview
  function previewCell(columnName, val) {
    if (columnName === "_flags") {
      return el("td", { class: val ? "flag-cell" : "" }, val ? el("span", { class: "val", text: val }) : null);
    }
    return el("td", {}, cell(val));
  }

  function previewPanel(r) {
    const views = { before: r.preview_before, after: r.preview_after };
    const holder = el("div");
    const bBefore = el("button", { type: "button", "aria-pressed": "false", text: "Before" });
    const bAfter = el("button", { type: "button", "aria-pressed": "true", text: "After" });

    function show(which) {
      bBefore.setAttribute("aria-pressed", String(which === "before"));
      bAfter.setAttribute("aria-pressed", String(which === "after"));
      const v = views[which];
      const head = el("thead", {}, el("tr", {}, v.columns.map((c) => el("th", { text: c }))));
      const body = el("tbody", {}, v.rows.map((row) => el("tr", {}, row.map((val, i) => previewCell(v.columns[i], val)))));
      holder.replaceChildren(
        el("div", { class: "table-wrap" }, el("table", { class: "preview" }, head, body)),
        el("p", { class: "note-small", text: `First ${v.rows.length} rows${which === "before" ? " exactly as uploaded." : " after cleaning."}` }));
    }
    bBefore.addEventListener("click", () => show("before"));
    bAfter.addEventListener("click", () => show("after"));
    show("after");
    return el("div", {}, el("div", { class: "seg" }, bBefore, bAfter), holder);
  }
})();
