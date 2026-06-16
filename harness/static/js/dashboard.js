// myharness web UI — client-side enhancements.
//
// Stays vanilla JS. No frameworks, no bundler, no build step. Operators
// can override this file via `dashboard.static_dir` (the /static/
// resolver checks the override dir first) without touching the wheel.
//
// What this script wires up:
//   1. enhanceTable()      — header-click sort + filter input for any
//                            <table id='…-table'> whose <th>s carry a
//                            data-sort attribute.
//   2. copyToClipboard()   — delegated click handler for any element
//                            with [data-copy]. Falls back to a textarea
//                            select on browsers without navigator.clipboard.
//   3. window.toast()      — toast helper, surfaced for PR-5 redirect
//                            success messages. Renders into #toast-host
//                            which is injected by _layout().
//   4. wireToastFromQuery() — read ?saved=…/?error=… from the URL on
//                            page load and surface a toast.
//
// Each helper guards on element existence so the script is a no-op when
// the page doesn't include the relevant markup.

(function () {
  "use strict";

  // -------------------------------------------------------------------
  // Toast surface
  // -------------------------------------------------------------------

  function ensureToastHost() {
    var host = document.getElementById("toast-host");
    if (host) return host;
    host = document.createElement("div");
    host.id = "toast-host";
    host.className = "toast-host";
    host.setAttribute("role", "status");
    host.setAttribute("aria-live", "polite");
    document.body.appendChild(host);
    return host;
  }

  function toast(message, level) {
    var host = ensureToastHost();
    var el = document.createElement("div");
    el.className = "toast toast--" + (level || "info");
    el.textContent = message;
    host.appendChild(el);
    // Slide-out after 4s.
    setTimeout(function () {
      el.classList.add("toast--leaving");
      setTimeout(function () { el.remove(); }, 200);
    }, 4000);
  }

  window.toast = toast;

  function wireToastFromQuery() {
    try {
      var params = new URLSearchParams(window.location.search);
      var saved = params.get("saved");
      var error = params.get("error");
      if (saved) toast("Saved " + saved + ".", "success");
      else if (error) toast(error, "error");
    } catch (_e) { /* old browser, ignore */ }
  }

  // -------------------------------------------------------------------
  // Copy-to-clipboard
  // -------------------------------------------------------------------

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand("copy");
        resolve();
      } catch (e) { reject(e); }
      finally { document.body.removeChild(ta); }
    });
  }

  function wireCopyButtons() {
    document.addEventListener("click", function (evt) {
      var btn = evt.target.closest("[data-copy]");
      if (!btn) return;
      evt.preventDefault();
      var text = btn.getAttribute("data-copy");
      if (!text) return;
      copyText(text).then(
        function () { toast("Copied", "success"); },
        function () { toast("Copy failed", "error"); }
      );
    });
  }

  // -------------------------------------------------------------------
  // Table sort + filter
  // -------------------------------------------------------------------

  function compareCells(a, b, kind) {
    if (kind === "num") {
      var na = parseFloat(a.replace(/[$,]/g, ""));
      var nb = parseFloat(b.replace(/[$,]/g, ""));
      if (isNaN(na) && isNaN(nb)) return 0;
      if (isNaN(na)) return 1;   // dashes / empty sort last
      if (isNaN(nb)) return -1;
      return na - nb;
    }
    if (kind === "date") {
      var da = Date.parse(a);
      var db = Date.parse(b);
      if (isNaN(da) && isNaN(db)) return 0;
      if (isNaN(da)) return 1;
      if (isNaN(db)) return -1;
      return da - db;
    }
    return a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
  }

  function sortTable(table, columnIndex, kind, direction) {
    var tbody = table.tBodies[0];
    if (!tbody) return;
    var rows = Array.prototype.slice.call(tbody.rows);
    rows.sort(function (r1, r2) {
      var c1 = r1.cells[columnIndex];
      var c2 = r2.cells[columnIndex];
      if (!c1 || !c2) return 0;
      var t1 = (c1.innerText || c1.textContent || "").trim();
      var t2 = (c2.innerText || c2.textContent || "").trim();
      return compareCells(t1, t2, kind) * direction;
    });
    rows.forEach(function (r) { tbody.appendChild(r); });
  }

  function enhanceTableSort(table) {
    var thead = table.tHead;
    if (!thead) return;
    var headers = thead.rows[0].cells;
    Array.prototype.forEach.call(headers, function (th, idx) {
      var kind = th.getAttribute("data-sort");
      if (!kind) return;
      th.classList.add("th-sortable");
      th.setAttribute("role", "button");
      th.setAttribute("tabindex", "0");
      var direction = 0; // 0=none, 1=asc, -1=desc
      function handle() {
        direction = direction === 1 ? -1 : 1;
        // Clear sibling indicators.
        Array.prototype.forEach.call(headers, function (sib) {
          sib.classList.remove("th-sorted-asc", "th-sorted-desc");
        });
        th.classList.add(direction === 1 ? "th-sorted-asc" : "th-sorted-desc");
        sortTable(table, idx, kind, direction);
      }
      th.addEventListener("click", handle);
      th.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); handle(); }
      });
    });
  }

  function enhanceTableFilter(table) {
    if (table.dataset.filter === "off") return;
    var wrap = table.parentElement;
    var card = wrap && wrap.closest(".card");
    // Render the filter input above the table-wrap.
    var input = document.createElement("input");
    input.type = "search";
    input.className = "bx--text-input table-filter";
    input.placeholder = "Filter " + (table.id ? table.id.replace(/-table$/, "") : "rows") + "…";
    input.setAttribute("aria-label", "Filter table rows");
    var holder = document.createElement("div");
    holder.className = "table-filter-row";
    holder.appendChild(input);
    wrap.parentNode.insertBefore(holder, wrap);
    var tbody = table.tBodies[0];
    if (!tbody) return;
    input.addEventListener("input", function () {
      var needle = input.value.trim().toLowerCase();
      var shown = 0;
      Array.prototype.forEach.call(tbody.rows, function (row) {
        var hay = (row.innerText || row.textContent || "").toLowerCase();
        var match = !needle || hay.indexOf(needle) !== -1;
        row.style.display = match ? "" : "none";
        if (match) shown++;
      });
      // Empty-result message.
      var emptyMsg = holder.querySelector(".table-filter-empty");
      if (shown === 0 && needle) {
        if (!emptyMsg) {
          emptyMsg = document.createElement("div");
          emptyMsg.className = "table-filter-empty muted mt-3";
          emptyMsg.textContent = "No rows match.";
          holder.appendChild(emptyMsg);
        }
      } else if (emptyMsg) {
        emptyMsg.remove();
      }
    });
  }

  function enhanceTables() {
    var tables = document.querySelectorAll("table[id$='-table']");
    Array.prototype.forEach.call(tables, function (t) {
      // Only enhance if it has a tbody (modern markup).
      if (!t.tBodies[0]) return;
      enhanceTableSort(t);
      enhanceTableFilter(t);
    });
  }

  // -------------------------------------------------------------------
  // Auto-refresh toggle (PR-5)
  // -------------------------------------------------------------------
  //
  // Polls by reloading the whole page on a fixed interval. Simpler than
  // a partial-fetch + innerHTML swap and works on every page without
  // server-side opt-in. The toggle state persists per origin in
  // localStorage so it survives reloads — that's the whole point.

  function wireAutoRefresh() {
    var btn = document.getElementById("auto-refresh-toggle");
    if (!btn) return;
    var STORAGE_KEY = "myharness:auto-refresh";
    var INTERVAL_MS = 15000;
    var timer = null;
    var label = btn.querySelector(".auto-refresh-btn__label");

    function setVisual(on) {
      btn.setAttribute("aria-pressed", on ? "true" : "false");
      if (label) label.textContent = on ? "Auto-refresh: on (15s)" : "Auto-refresh: off";
    }
    function start() {
      stop();
      timer = window.setInterval(function () { window.location.reload(); }, INTERVAL_MS);
    }
    function stop() {
      if (timer) { window.clearInterval(timer); timer = null; }
    }

    btn.addEventListener("click", function () {
      var on = btn.getAttribute("aria-pressed") !== "true";
      setVisual(on);
      try {
        if (on) localStorage.setItem(STORAGE_KEY, "1");
        else localStorage.removeItem(STORAGE_KEY);
      } catch (_e) { /* private mode etc. */ }
      if (on) start(); else stop();
    });

    // Restore last state.
    try {
      if (localStorage.getItem(STORAGE_KEY) === "1") { setVisual(true); start(); }
    } catch (_e) { /* ignore */ }
  }

  // -------------------------------------------------------------------
  // Session purge (Resume picker: × Delete column)
  // -------------------------------------------------------------------
  //
  // Each row's Delete button is a plain <button data-purge-session=…>
  // OUTSIDE any form — clicking it confirms with the operator, then
  // POSTs to /sessions/{sid}/purge with the CSRF token from the cookie
  // in an X-CSRF-Token header. On success we navigate to the redirect
  // target the server returned (typically /run?mode=resume&saved=…).

  function readCookie(name) {
    var parts = (document.cookie || "").split(";");
    for (var i = 0; i < parts.length; i++) {
      var kv = parts[i].split("=");
      if (kv[0] && kv[0].trim() === name) {
        return decodeURIComponent((kv[1] || "").trim());
      }
    }
    return "";
  }

  function wireSessionPurge() {
    document.addEventListener("click", function (evt) {
      var btn = evt.target.closest("[data-purge-session]");
      if (!btn) return;
      evt.preventDefault();
      evt.stopPropagation();
      var sid = btn.getAttribute("data-purge-session");
      if (!sid) return;
      var msg = "Permanently delete session " + sid +
                "?\n\nThis wipes the checkpoint rows from the SQLite store " +
                "AND removes the JSONL log on disk. Cannot be undone.";
      if (!window.confirm(msg)) return;

      var csrf = readCookie("csrf_token");
      btn.disabled = true;
      btn.classList.add("ct-remove--busy");

      fetch("/sessions/" + encodeURIComponent(sid) + "/purge", {
        method: "POST",
        credentials: "same-origin",
        redirect: "manual",   // we want to see the 303 location ourselves
        headers: {
          "X-CSRF-Token": csrf,
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: "",
      }).then(function (resp) {
        // The server returns 303 + Location; with redirect:manual the
        // fetch resolves to an opaque-redirect (type=opaqueredirect)
        // which we can't read headers from. Just navigate to the
        // canonical post-delete URL ourselves.
        if (resp.type === "opaqueredirect" || resp.status === 303 ||
            resp.status === 200 || resp.status === 0) {
          window.location.href = "/run?mode=resume&saved=" +
            encodeURIComponent("deleted session " + sid);
          return;
        }
        return resp.text().then(function (text) {
          throw new Error((text || "purge failed").trim().slice(0, 240));
        });
      }).catch(function (err) {
        toast("Delete failed: " + err.message, "error");
        btn.disabled = false;
        btn.classList.remove("ct-remove--busy");
      });
    });
  }

  // -------------------------------------------------------------------
  // /run?mode=resume — preselect the Resume tab on page load
  // -------------------------------------------------------------------
  //
  // The Run page's mode tabs are pure-CSS radios with the New radio
  // checked by default. After a delete (or any link that wants to
  // land on the Resume view), append ?mode=resume to the URL and this
  // helper flips the radio on load.

  function wireRunModeFromQuery() {
    var mode;
    try {
      mode = new URLSearchParams(window.location.search).get("mode");
    } catch (_e) { return; }
    if (mode !== "resume") return;
    var radio = document.getElementById("mode-resume");
    if (radio) {
      radio.checked = true;
      var ev = new Event("change", { bubbles: true });
      radio.dispatchEvent(ev);
    }
  }

  // -------------------------------------------------------------------
  // Session picker row-click → toggle the row's radio
  // -------------------------------------------------------------------
  //
  // Clicking anywhere in a session-picker row selects that session.
  // The label-wrapped session id already makes the id cell clickable;
  // this handler extends the same affordance to the rest of the row
  // so operators don't have to aim for the small radio.

  function wireSessionPicker() {
    document.addEventListener("click", function (evt) {
      var row = evt.target.closest(".session-picker__table tbody tr");
      if (!row) return;
      // Ignore clicks that originated on form controls or links — those
      // have their own behavior.
      var tag = evt.target.tagName;
      if (tag === "INPUT" || tag === "LABEL" || tag === "A" || tag === "BUTTON") {
        return;
      }
      var radio = row.querySelector('input[type="radio"][name="resume_session_id"]');
      if (radio && !radio.checked) {
        radio.checked = true;
        // Fire a change event so any listeners (and CSS :checked
        // selectors that need the change pulse) update.
        var ev = new Event("change", { bubbles: true });
        radio.dispatchEvent(ev);
      }
    });
  }

  // -------------------------------------------------------------------
  // Mobile nav toggle (PR-6)
  // -------------------------------------------------------------------

  function wireNavToggle() {
    var btn = document.getElementById("nav-toggle");
    if (!btn) return;
    function setOpen(open) {
      btn.setAttribute("aria-expanded", open ? "true" : "false");
      if (open) document.body.dataset.navOpen = "1";
      else delete document.body.dataset.navOpen;
    }
    btn.addEventListener("click", function () {
      setOpen(document.body.dataset.navOpen !== "1");
    });
    // Click on the backdrop (anywhere outside the nav) closes it.
    document.addEventListener("click", function (e) {
      if (document.body.dataset.navOpen !== "1") return;
      var nav = document.getElementById("side-nav");
      if (!nav) return;
      if (nav.contains(e.target) || btn.contains(e.target)) return;
      setOpen(false);
    });
    // ESC closes too.
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && document.body.dataset.navOpen === "1") {
        setOpen(false);
      }
    });
  }

  // -------------------------------------------------------------------
  // Live event stream filter (PR-6)
  // -------------------------------------------------------------------
  //
  // Opt-in via <ul id="event-stream" data-sse-url="/api/...events">.
  // The renderer also provides a <div class="event-stream-filters">
  // with one chip per known event type; clicking a chip toggles its
  // CSS visibility. This replaces the legacy <pre id="live-events">
  // dump which had no filtering and no parsing.

  function wireEventStream() {
    var ul = document.getElementById("event-stream");
    if (!ul || typeof EventSource === "undefined") return;
    var url = ul.getAttribute("data-sse-url");
    if (!url) return;
    var es = new EventSource(url);
    es.onmessage = function (evt) {
      var data;
      try { data = JSON.parse(evt.data); } catch (_e) { data = { raw: evt.data }; }
      var kind = (data && data.event) || "unknown";
      var ts = (data && data.timestamp) || "";
      var rest = Object.assign({}, data);
      delete rest.event;
      delete rest.timestamp;
      var li = document.createElement("li");
      li.setAttribute("data-event-type", kind);
      var head = document.createElement("div");
      head.className = "event-stream__head";
      head.innerHTML =
        '<span class="event-stream__type tag tag-blue">' + escapeText(kind) + "</span>" +
        '<span class="event-stream__ts muted fs-sm">' + escapeText(ts) + "</span>";
      var body = document.createElement("pre");
      body.className = "event-stream__body";
      body.textContent = JSON.stringify(rest, null, 2);
      li.appendChild(head);
      li.appendChild(body);
      ul.insertBefore(li, ul.firstChild);
      // Cap to 500 items in the DOM to avoid runaway memory.
      while (ul.children.length > 500) ul.removeChild(ul.lastChild);
      registerFilter(kind);
    };
    es.addEventListener("close", function () { es.close(); });
  }

  function escapeText(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  var seenEventTypes = {};
  function registerFilter(kind) {
    if (seenEventTypes[kind]) return;
    seenEventTypes[kind] = true;
    var bar = document.querySelector(".event-stream-filters");
    if (!bar) return;
    var chip = document.createElement("button");
    chip.type = "button";
    chip.className = "event-chip event-chip--on";
    chip.setAttribute("data-filter", kind);
    chip.setAttribute("aria-pressed", "true");
    chip.textContent = kind;
    chip.addEventListener("click", function () {
      var on = chip.classList.toggle("event-chip--on");
      chip.setAttribute("aria-pressed", on ? "true" : "false");
      var ul = document.getElementById("event-stream");
      if (!ul) return;
      Array.prototype.forEach.call(
        ul.querySelectorAll('li[data-event-type="' + kind + '"]'),
        function (li) { li.style.display = on ? "" : "none"; }
      );
    });
    bar.appendChild(chip);
  }

  // -------------------------------------------------------------------
  // Config tree — + Add / × Remove for editable collections
  // -------------------------------------------------------------------
  //
  // The Configure Harness page renders config.json as a nested form.
  // Collections (dict_record / list_record / list_scalar / dict_scalar)
  // each get a + Add button and per-entry × Remove buttons. This
  // handler is fully delegated so dynamically-inserted rows pick up
  // the wiring without re-init.

  function wireConfigTree() {
    document.addEventListener("click", function (evt) {
      var addBtn = evt.target.closest(".ct-add");
      if (addBtn) {
        evt.preventDefault();
        handleAdd(addBtn);
        return;
      }
      var removeBtn = evt.target.closest(".ct-remove");
      if (removeBtn) {
        evt.preventDefault();
        handleRemove(removeBtn);
        return;
      }
    });
  }

  function handleRemove(btn) {
    var target = btn.getAttribute("data-target");
    if (target === "record") {
      var rec = btn.closest("details.ct-record");
      if (rec) rec.remove();
    } else if (target === "row") {
      var row = btn.closest(".ct-row");
      if (row) row.remove();
    } else if (target === "item") {
      var item = btn.closest(".ct-list__item");
      if (item) item.remove();
    }
  }

  function handleAdd(btn) {
    var kind = btn.getAttribute("data-collection");
    var path = btn.getAttribute("data-path") || "";
    var container = btn.parentElement;
    if (kind === "dict_record") {
      addDictRecord(container, path);
    } else if (kind === "list_record") {
      addListRecord(container, path);
    } else if (kind === "list_scalar") {
      addListScalar(container, path, btn.getAttribute("data-type") || "str");
    } else if (kind === "dict_scalar") {
      addDictScalar(container, path, btn.getAttribute("data-type") || "str");
    }
  }

  function addDictRecord(container, path) {
    // Clone the record template and insert IMMEDIATELY — no upfront
    // key prompt. The new record carries an editable key input in its
    // header so the operator can name it inline after the schema is
    // visible. Child __path[] inputs auto-update as the key changes.
    var template = container.querySelector(".ct-template");
    if (!template) {
      toast("No template registered for this collection.", "error");
      return;
    }
    // Generate a placeholder key that doesn't collide with existing
    // records ("new-1", "new-2", ...). The operator renames inline.
    var existing = container.querySelectorAll(":scope > details.ct-record");
    var taken = {};
    Array.prototype.forEach.call(existing, function (el) {
      var k = el.getAttribute("data-dict-key");
      if (k) taken[k] = true;
    });
    var placeholder = "new-1";
    var n = 1;
    while (taken[placeholder]) { n++; placeholder = "new-" + n; }

    var newRecHtml = template.innerHTML.replace(/__NEW_KEY__/g, placeholder);
    var holder = document.createElement("div");
    holder.innerHTML = newRecHtml.trim();
    var newRec = holder.firstElementChild;
    if (!newRec) return;
    container.insertBefore(newRec, template);

    // Wire the key editor on the new record so the operator can rename
    // it; the JS keeps child __path[] inputs in sync.
    attachKeyEditor(newRec, placeholder, path);

    // Focus the key input and select its text so the next keystroke
    // replaces "new-1" with the operator's real name.
    var keyInput = newRec.querySelector(".ct-record__key-input");
    if (keyInput) {
      keyInput.focus();
      try { keyInput.select(); } catch (_e) { /* ok */ }
    }
    toast("Added — name it and fill the fields", "success");
  }

  function attachKeyEditor(record, initialKey, parentPath) {
    var input = record.querySelector(".ct-record__key-input");
    if (!input) return;
    var currentKey = initialKey;
    input.addEventListener("input", function () {
      var newKey = input.value.trim();
      if (!newKey || newKey === currentKey) return;
      // Disallow keys that collide with sibling records.
      var parent = record.parentElement;
      var siblingMatch = parent && parent.querySelector(
        'details.ct-record[data-dict-key="' + cssEscape(newKey) + '"]'
      );
      if (siblingMatch && siblingMatch !== record) {
        input.setCustomValidity("Key already exists");
        return;
      }
      input.setCustomValidity("");
      // Update the record's identity attribute + every __path[] under it.
      var oldPrefix = parentPath + "/" + currentKey + "/";
      var newPrefix = parentPath + "/" + newKey + "/";
      var paths = record.querySelectorAll('input[name="__path[]"]');
      Array.prototype.forEach.call(paths, function (pi) {
        if (pi.value.indexOf(oldPrefix) === 0) {
          pi.value = newPrefix + pi.value.substring(oldPrefix.length);
        }
      });
      record.setAttribute("data-dict-key", newKey);
      currentKey = newKey;
    });
    // Pressing Enter shouldn't submit the form — operators expect it
    // to commit the rename and move on.
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") { e.preventDefault(); input.blur(); }
    });
  }

  function addListRecord(container, path) {
    var template = container.querySelector(".ct-template");
    if (!template) {
      toast("No template to add from.", "error");
      return;
    }
    // Compute next index by counting existing records.
    var existing = container.querySelectorAll(":scope > details.ct-record");
    var nextIndex = existing.length;
    var newRecHtml = template.innerHTML.replace(/__INDEX__/g, String(nextIndex));
    var holder = document.createElement("div");
    holder.innerHTML = newRecHtml.trim();
    var newRec = holder.firstElementChild;
    if (!newRec) return;
    container.insertBefore(newRec, template);
    toast("Added record #" + nextIndex, "success");
  }

  function addListScalar(container, path, typeTag) {
    var existing = container.querySelectorAll(":scope > .ct-list__item");
    var nextIndex = existing.length;
    var fullPath = path + "/" + nextIndex;
    var item = document.createElement("div");
    item.className = "ct-list__item";
    item.innerHTML =
      '<span class="ct-list__index">' + nextIndex + "</span>" +
      '<div class="ct-list__value">' +
        '<input type="hidden" name="__path[]" value="' + escapeAttr(fullPath) + '">' +
        '<input type="hidden" name="__type[]" value="' + escapeAttr(typeTag) + '">' +
        scalarInputHtml(typeTag, "") +
      "</div>" +
      '<button type="button" class="ct-remove" data-target="item" aria-label="Remove entry">&times;</button>';
    var addBtn = container.querySelector(".ct-add");
    container.insertBefore(item, addBtn);
  }

  function addDictScalar(container, path, typeTag) {
    var keyInput = container.querySelector(".ct-add-row .ct-new-key");
    var newKey = (keyInput && keyInput.value || "").trim();
    if (!newKey) {
      toast("Pick a key name first", "error");
      if (keyInput) keyInput.focus();
      return;
    }
    var fullPath = path + "/" + newKey;
    var row = document.createElement("div");
    row.className = "ct-row ct-row--dict-entry";
    row.innerHTML =
      '<label class="bx--label ct-row__label">' +
        escapeText(prettyKey(newKey)) +
        '<code class="ct-row__key" aria-hidden="true">' + escapeText(newKey) + "</code>" +
      "</label>" +
      '<div class="ct-row__value">' +
        '<input type="hidden" name="__path[]" value="' + escapeAttr(fullPath) + '">' +
        '<input type="hidden" name="__type[]" value="' + escapeAttr(typeTag) + '">' +
        scalarInputHtml(typeTag, "") +
      "</div>" +
      '<button type="button" class="ct-remove" data-target="row" aria-label="Remove entry">&times;</button>';
    var addBlock = container.querySelector(".ct-add-row");
    container.insertBefore(row, addBlock);
    if (keyInput) keyInput.value = "";
  }

  function scalarInputHtml(typeTag, value) {
    if (typeTag === "bool") {
      return (
        '<input type="hidden" name="__value[]" value="false">' +
        '<input type="checkbox" class="ct-bool" name="__value[]" value="true">'
      );
    }
    if (typeTag === "int") {
      return (
        '<input type="number" step="1" class="bx--text-input ct-input" ' +
        'name="__value[]" value="' + escapeAttr(value) + '">'
      );
    }
    if (typeTag === "float") {
      return (
        '<input type="number" step="any" class="bx--text-input ct-input" ' +
        'name="__value[]" value="' + escapeAttr(value) + '">'
      );
    }
    return (
      '<input type="text" class="bx--text-input ct-input" ' +
      'name="__value[]" value="' + escapeAttr(value) + '">'
    );
  }

  function prettyKey(k) {
    if (!k) return "";
    var s = k.replace(/_/g, " ");
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

  function escapeAttr(s) {
    return escapeText(s).replace(/"/g, "&quot;");
  }
  function cssEscape(s) {
    if (window.CSS && window.CSS.escape) return CSS.escape(s);
    return s.replace(/[^a-zA-Z0-9_-]/g, "\\$&");
  }

  // -------------------------------------------------------------------
  // Boot
  // -------------------------------------------------------------------

  function boot() {
    ensureToastHost();
    wireCopyButtons();
    enhanceTables();
    wireAutoRefresh();
    wireNavToggle();
    wireEventStream();
    wireConfigTree();
    wireSessionPicker();
    wireSessionPurge();
    wireRunModeFromQuery();
    wireToastFromQuery();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
