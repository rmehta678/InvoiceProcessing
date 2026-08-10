/* Galatiq Invoice Console - progressive enhancement only.
   Everything here is presentation: copy buttons, decision sub-form visibility,
   keyboard navigation, and the SSE live timeline. All state shown comes from
   the server; nothing is computed into a status client-side. */

(function () {
  "use strict";

  /* ---------------------------------------------- HTMX mutation protection */

  document.body.addEventListener("htmx:configRequest", function (event) {
    var verb = String(event.detail.verb || "GET").toUpperCase();
    if (verb === "GET" || verb === "HEAD" || verb === "OPTIONS") return;
    if (event.detail.parameters && event.detail.parameters.csrf_token !== undefined) return;
    var token = document.querySelector("meta[name=csrf-token]");
    if (token && token.content) event.detail.headers["X-CSRF-Token"] = token.content;
  });

  /* ------------------------------------------------------------ copy buttons */

  document.addEventListener("click", function (event) {
    var button = event.target.closest("button.copy");
    if (!button) return;
    var value = button.getAttribute("data-copy") || "";
    navigator.clipboard.writeText(value).then(function () {
      button.classList.add("copied");
      var previous = button.textContent;
      button.textContent = "copied";
      window.setTimeout(function () {
        button.classList.remove("copied");
        button.textContent = previous;
      }, 1200);
    });
  });

  /* ------------------------------------------------------ select-all boxes */

  document.querySelectorAll("input[data-select-all]").forEach(function (master) {
    var table = master.closest("table");
    if (!table) return;
    function boxes() {
      return Array.prototype.slice.call(
        table.querySelectorAll("tbody input[type=checkbox]")
      );
    }
    master.addEventListener("change", function () {
      boxes().forEach(function (box) { box.checked = master.checked; });
    });
    table.addEventListener("change", function (event) {
      if (event.target === master || event.target.type !== "checkbox") return;
      var all = boxes();
      var checked = all.filter(function (box) { return box.checked; });
      master.checked = all.length > 0 && checked.length === all.length;
      master.indeterminate = checked.length > 0 && checked.length < all.length;
    });
  });

  /* ------------------------------------------- decision sub-form visibility */

  function syncSubforms(form) {
    var checked = form.querySelector("input[name=decision]:checked");
    form.querySelectorAll(".subform").forEach(function (block) {
      var owner = block.getAttribute("data-for");
      var active = !!checked && owner === checked.value;
      block.classList.toggle("visible", active);
      block.querySelectorAll("input, select, textarea, button").forEach(function (control) {
        control.disabled = !active;
      });
    });
  }

  document.querySelectorAll("form[data-decision-form]").forEach(function (form) {
    form.addEventListener("change", function (event) {
      if (event.target.name === "decision") syncSubforms(form);
    });
    syncSubforms(form);
  });

  /* --------------------------------------------------- keyboard navigation */

  function rowLinks() {
    return Array.prototype.slice.call(
      document.querySelectorAll("table.data tbody tr[data-href] a[data-row-link]")
    );
  }

  function setKbFocus(links, index) {
    links.forEach(function (link) {
      var row = link.closest("tr");
      if (row) row.classList.remove("kb-focus");
    });
    if (index >= 0 && index < links.length) {
      var link = links[index];
      var row = link.closest("tr");
      if (row) row.classList.add("kb-focus");
      link.focus();
      link.scrollIntoView({ block: "nearest", inline: "nearest" });
    }
  }

  document.addEventListener("keydown", function (event) {
    var tag = (document.activeElement && document.activeElement.tagName) || "";
    var typing = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
    if (event.key === "/" && !typing) {
      var search = document.querySelector("input[type=search]");
      if (search) {
        event.preventDefault();
        search.focus();
        search.select();
      }
      return;
    }
    if (typing) return;
    var links = rowLinks();
    if (!links.length) return;
    var activeIndex = links.indexOf(document.activeElement);
    if (event.key === "j") {
      event.preventDefault();
      setKbFocus(links, Math.min(activeIndex + 1, links.length - 1));
    } else if (event.key === "k") {
      event.preventDefault();
      setKbFocus(links, Math.max(activeIndex - 1, 0));
    }
  });

  document.addEventListener("click", function (event) {
    var row = event.target.closest("tr[data-href]");
    if (!row) return;
    var interactive = [
      "a",
      "area",
      "button",
      "input",
      "select",
      "textarea",
      "label",
      "form",
      "details",
      "summary",
      "audio[controls]",
      "video[controls]",
      "iframe",
      "embed",
      "object",
      "[contenteditable]",
      "[tabindex]",
      "[role]"
    ].join(", ");
    if (
      event.defaultPrevented ||
      event.target.closest(interactive)
    ) return;
    var link = row.querySelector("a[data-row-link]");
    if (link) link.click();
  });

  /* ------------------------------------------------------- live SSE timeline */

  var live = document.querySelector("[data-live-events]");
  if (!live) return;

  var caseId = live.getAttribute("data-case-id");
  var list = document.getElementById("live-timeline");
  var note = document.getElementById("stream-note");
  var bannerHost = document.getElementById("terminal-host");
  var toolStarts = {}; /* call id -> request timestamp (server created_at) */

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function shortTime(iso) {
    var match = /T(\d\d:\d\d:\d\d)/.exec(iso || "");
    return match ? match[1] : (iso || "");
  }

  function agentChip(name) {
    return el("span", "chip agent", name || "system");
  }

  function appendRow(data) {
    var row = el("li");
    row.appendChild(el("span", "ts", shortTime(data.created_at)));
    var type = data.event_type || "";
    if (data.handoff) {
      row.appendChild(agentChip(data.handoff.source));
      row.appendChild(el("span", "arrow", "\u2192"));
      row.appendChild(agentChip(data.handoff.target));
      row.appendChild(el("span", "dim", "handoff"));
    } else if (data.tool_calls) {
      row.appendChild(agentChip(data.agent));
      data.tool_calls.forEach(function (call) {
        if (call.id) toolStarts[call.id] = data.created_at;
        row.appendChild(el("span", "chip locator", (call.name || "tool") + "()"));
      });
      row.appendChild(el("span", "dim", "tool call"));
    } else if (data.tool_results) {
      row.appendChild(agentChip(data.agent));
      var anyError = false;
      data.tool_results.forEach(function (result) {
        var label = (result.name || "tool") + " done";
        var started = result.id && toolStarts[result.id];
        if (started) {
          var ms = Date.parse(data.created_at) - Date.parse(started);
          if (!isNaN(ms) && ms >= 0) label += " in " + (ms / 1000).toFixed(1) + "s";
        }
        if (result.is_error) { label += " (error)"; anyError = true; }
        row.appendChild(el("span", "chip locator", label));
      });
      if (anyError) row.classList.add("tool-error");
    } else if (type === "provider.retry") {
      row.classList.add("retry");
      row.appendChild(el("span", null, "provider retry"));
      if (data.message) row.appendChild(el("span", "dim", data.message));
    } else {
      if (data.agent) row.appendChild(agentChip(data.agent));
      row.appendChild(el("span", "dim", type));
      if (data.status) {
        row.appendChild(el("span", "mono", data.status + " / " + (data.stop_reason || "")));
      }
    }
    list.appendChild(row);
    row.scrollIntoView({ block: "nearest" });
  }

  var toneByStatus = {
    SUCCEEDED: "tone-ok",
    NEEDS_HUMAN: "tone-warn",
    FAILED: "tone-fail",
    INCOMPLETE: "tone-pause"
  };

  function showTerminal(data) {
    if (note) note.textContent = "";
    var banner = el("div", "terminal-banner " + (toneByStatus[data.status] || "tone-pause"));
    if (data.missing) {
      banner.className = "terminal-banner tone-fail";
      banner.appendChild(el("div", null, "This case does not exist in the workflow database."));
    } else {
      banner.appendChild(el("div", null, (data.status || "?") + " - " + (data.stop_reason || "")));
      if (data.run_error) {
        banner.appendChild(el("div", "small", "run error: " + data.run_error));
      }
      var link = el("a", null, "Open case detail (stored state)");
      link.href = "/cases/" + encodeURIComponent(caseId);
      banner.appendChild(link);
    }
    bannerHost.textContent = "";
    bannerHost.appendChild(banner);
  }

  function showRecoveryError(data) {
    if (note) {
      note.textContent = "Execution recovery unavailable. Stored terminal state was not verified.";
      note.classList.add("dropped");
      note.classList.remove("pulsing");
    }
    var banner = el("div", "terminal-banner tone-fail");
    banner.appendChild(el("div", null, "Execution recovery unavailable"));
    banner.appendChild(
      el("div", "small mono", data.stop_reason || "EXECUTION_RECOVERY_FAILED")
    );
    var link = el("a", null, "Open case detail (stored state)");
    link.href = "/cases/" + encodeURIComponent(caseId);
    banner.appendChild(link);
    bannerHost.textContent = "";
    bannerHost.appendChild(banner);
  }

  var source = new EventSource("/cases/" + encodeURIComponent(caseId) + "/events");
  source.addEventListener("case-event", function (event) {
    appendRow(JSON.parse(event.data));
  });
  source.addEventListener("terminal", function (event) {
    showTerminal(JSON.parse(event.data));
    source.close();
    if (note) note.classList.remove("pulsing");
  });
  source.addEventListener("recovery-error", function (event) {
    source.close();
    var data;
    try {
      data = JSON.parse(event.data);
    } catch (_error) {
      data = { stop_reason: "EXECUTION_RECOVERY_FAILED" };
    }
    showRecoveryError(data);
  });
  source.onerror = function () {
    if (note) {
      note.textContent =
        "Event stream dropped. The database remains the source of truth - open the case detail below for persisted state.";
      note.classList.add("dropped");
      note.classList.remove("pulsing");
    }
  };
})();
