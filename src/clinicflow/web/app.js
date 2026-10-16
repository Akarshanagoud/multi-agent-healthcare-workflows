/* Clinical Review Desk: reviewer-facing views over the clinicflow workflow. */
(function () {
  "use strict";
  const { h, api, fmt } = Kit;

  const DECISION = {
    complete: { kind: "good", label: "Complete", color: "var(--good)" },
    needs_review: { kind: "warning", label: "Needs review", color: "var(--warning)" },
    rejected: { kind: "critical", label: "Rejected", color: "var(--critical)" },
    failed: { kind: "serious", label: "Failed", color: "var(--serious)" },
  };
  const DECISION_ORDER = ["complete", "needs_review", "rejected", "failed"];
  const SEVERITY = { info: "", warning: "warning", error: "serious", critical: "critical" };
  const SEVERITY_RANK = { info: 0, warning: 1, error: 2, critical: 3 };
  const AGENT_NOTE = {
    intake: "PHI minimisation and injection screening",
    router: "classifies the document and picks the route",
    retriever: "finds the governing policy passages",
    extractor: "rules first, model second, spans always",
    validator: "grounding, ranges, interactions, completeness",
    summarizer: "assembled from verified fields only",
  };

  let samplesCache = null;
  const samples = async () => samplesCache || (samplesCache = await api.get("/v1/ui/samples"));

  const decisionBadge = (d) => Kit.badge((DECISION[d] || { label: d }).label, (DECISION[d] || {}).kind);
  const severityBadge = (s) => Kit.badge(fmt.title(s), SEVERITY[s] || "", { icon: s !== "info" });
  const docType = (t) => (t ? fmt.title(t) : "—");
  const FIELD_LABEL = { mrn: "MRN", icd10_code: "ICD-10 code", npi: "NPI", hba1c: "HbA1c", dob: "Date of birth", spo2: "SpO2", cpt_code: "CPT code", wbc: "WBC" };
  const fieldLabel = (name) => FIELD_LABEL[name] || fmt.title(name);

  function fieldValue(f) {
    const v = f.value;
    const num = (x) => (typeof x === "number" ? String(Number(x.toFixed(2))) : x);
    if (Array.isArray(v)) {
      return v
        .map((item) => (item && typeof item === "object" ? [item.drug || item.name, num(item.dose), item.unit, item.route, item.frequency].filter((x) => x !== undefined && x !== null && x !== "").join(" ") : String(item)))
        .join(", ");
    }
    if (v && typeof v === "object") return Object.entries(v).map(([k, x]) => k + " " + num(x)).join(", ");
    return String(num(v)) + (f.unit ? " " + f.unit : "");
  }

  function confidence(c) {
    return h("span", { class: "conf" }, Kit.meter(c, c >= 0.8 ? "good" : c >= 0.6 ? "warning" : "critical", "Confidence"), h("span", { class: "num small" }, fmt.num(c, 2)));
  }

  // ------------------------------------------------------------------ queue
  async function queueView(view) {
    const rerun = h("button", { class: "btn", type: "button" }, "Re-run workflow");
    const data = await api.post("/v1/ui/queue");
    rerun.addEventListener("click", () => app.refresh());
    const s = data.summary;
    const rows = data.rows;
    const matched = rows.filter((r) => r.matches_expected).length;
    const flagged = rows.filter((r) => r.decision !== "complete").length;
    view.append(
      Kit.pageHead("Review queue", "The labelled sample set through the full multi-agent workflow. Anything a clinician must see is at the top, worst first.", [rerun]),
      h(
        "div",
        { class: "grid cols-5" },
        Kit.stat({ label: "Documents", value: fmt.int(s.documents), note: fmt.ms(s.mean_duration_ms) + " mean processing time" }),
        Kit.stat({ label: "Straight through", value: fmt.pct(s.straight_through_rate, 0), note: (s.decisions.complete || 0) + " completed without review" }),
        Kit.stat({ label: "Waiting for a clinician", value: fmt.int(flagged), note: (s.decisions.needs_review || 0) + " review · " + (s.decisions.rejected || 0) + " quarantined" }),
        Kit.stat({ label: "Mean confidence", value: fmt.num(s.mean_confidence, 2), note: "across extracted fields" }),
        Kit.stat({ label: "Matches the label", value: matched + " / " + rows.length, note: "decision equals the expected outcome" }),
      ),
    );
    const decisions = DECISION_ORDER.filter((d) => s.decisions[d]).map((d) => ({ label: DECISION[d].label, value: s.decisions[d], color: DECISION[d].color }));
    const issues = s.top_issues.map(([code, n]) => ({ label: code, value: n }));
    const tableCard = Kit.card({
      title: "Queue",
      subtitle: "Click a document to review its fields, evidence and findings",
      flush: true,
      body: Kit.table({
        columns: [
          { key: "doc_id", label: "Document", render: (r) => h("div", null, h("b", null, r.doc_id), h("div", { class: "small muted" }, r.label)) },
          { key: "doc_type", label: "Type", cls: "hide-sm", render: (r) => docType(r.doc_type) },
          { key: "decision", label: "Decision", render: (r) => decisionBadge(r.decision), sortValue: (r) => DECISION_ORDER.indexOf(r.decision) },
          { key: "max_severity", label: "Worst finding", cls: "hide-sm", render: (r) => severityBadge(r.max_severity), sortValue: (r) => r.severity_rank },
          { key: "confidence", label: "Confidence", render: (r) => confidence(r.confidence) },
          { key: "reasons", label: "Why it is here", sortable: false, cls: "hide-sm", render: (r) => (r.reasons.length ? h("span", { class: "small" }, r.reasons[0], r.reasons.length > 1 ? h("span", { class: "muted" }, " +" + (r.reasons.length - 1) + " more") : null) : h("span", { class: "muted small" }, "no findings")) },
          { key: "matches_expected", label: "Label", align: "center", render: (r) => (r.matches_expected ? Kit.badge("Match", "good") : Kit.badge("Mismatch", "critical", { title: "Expected " + r.expected_decision })) },
        ],
        rows,
        onRowClick: (r) => app.go("review", { doc: r.doc_id }),
      }),
    });
    view.append(
      h(
        "div",
        { class: "grid cols-2 mt" },
          Kit.chartCard({
            title: "Decisions",
            subtitle: "How the batch was routed",
            chart: () => Kit.hbars(decisions, { format: fmt.int, valueLabel: "Documents" }),
            table: () => ({ columns: [{ key: "label", label: "Decision" }, { key: "value", label: "Documents", align: "right" }], rows: decisions }),
          }),
          Kit.chartCard({
            title: "Most frequent findings",
            subtitle: "Issue codes raised across the batch",
            chart: () => Kit.hbars(issues, { format: fmt.int, valueLabel: "Raised" }),
            table: () => ({ columns: [{ key: "label", label: "Code" }, { key: "value", label: "Raised", align: "right" }], rows: issues }),
          }),
      ),
      h("div", { class: "mt" }, tableCard),
    );
  }

  // ------------------------------------------------------------------ review
  function documentPanel(r) {
    const spans = Object.values(r.fields)
      .filter((f) => f.span && r.agent_text.slice(f.span[0], f.span[1]))
      .map((f) => {
        const flagged = r.issues.some((i) => i.field === f.name && SEVERITY_RANK[i.severity] >= 1);
        return { start: f.span[0], end: f.span[1], cls: flagged ? "m-critical" : "m-info", title: fieldLabel(f.name) + ": " + fieldValue(f), id: f.name };
      });
    const body = h("div", { class: "doc", id: "doc-text" });
    let showing = "agent";
    const paint = () => {
      Kit.clear(body);
      if (showing === "agent") body.append(Kit.highlight(r.agent_text, spans));
      else body.append(r.original_text);
    };
    paint();
    const toggle =
      r.agent_text !== r.original_text
        ? Kit.seg(
            [
              { value: "agent", label: "What the agents read" },
              { value: "original", label: "Original" },
            ],
            "agent",
            (v) => {
              showing = v;
              paint();
            },
          )
        : null;
    return Kit.card({
      title: "Document",
      subtitle: r.decision === "rejected" ? "Quarantined at intake: no agent processed it" : "Highlights mark the text each field was extracted from; red marks a field with a finding",
      actions: toggle,
      body: h(
        "div",
        null,
        body,
        r.phi_entities.length
          ? h("div", { class: "phi-note" }, "Removed or pseudonymised before any agent saw it:", r.phi_entities.map((e) => Kit.badge(e, "info", { icon: false })))
          : null,
      ),
    });
  }

  function focusSpan(name) {
    document.querySelectorAll("#doc-text mark.is-active").forEach((m) => m.classList.remove("is-active"));
    const mark = document.querySelector('#doc-text mark[data-span-id="' + CSS.escape(name) + '"]');
    if (mark) {
      mark.classList.add("is-active");
      mark.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  }

  function renderReview(host, r, label) {
    Kit.clear(host);
    const d = DECISION[r.decision] || { kind: "" };
    const findings = r.issues.slice().sort((a, b) => SEVERITY_RANK[b.severity] - SEVERITY_RANK[a.severity]);
    const titles = {
      complete: "Complete: no clinician review required",
      needs_review: "Needs review: " + r.review_reasons.length + " finding" + (r.review_reasons.length === 1 ? "" : "s") + " for a clinician",
      rejected: "Rejected: the document contains text addressed to the model",
      failed: "Failed: the workflow could not finish",
    };
    host.append(
      Kit.banner({
        kind: d.kind,
        title: titles[r.decision] || r.decision,
        text: [label, docType(r.doc_type), "confidence " + fmt.num(r.mean_confidence, 2), fmt.ms(r.duration_ms), "run " + fmt.short(r.run_id)].filter(Boolean).join(" · "),
        extra: r.review_reasons.length ? h("ul", { style: { margin: "6px 0 0", paddingLeft: "18px" } }, r.review_reasons.map((x) => h("li", { class: "small" }, x))) : null,
      }),
    );
    const fieldRows = Object.values(r.fields).sort((a, b) => (a.span ? a.span[0] : 1e9) - (b.span ? b.span[0] : 1e9));
    const fieldsCard = Kit.card({
      title: "Extracted fields",
      subtitle: fieldRows.length + " fields, each tied to the text it came from",
      flush: true,
      body: Kit.table({
        columns: [
          { key: "name", label: "Field", render: (f) => fieldLabel(f.name) },
          { key: "value", label: "Value", sortable: false, render: (f) => fieldValue(f) },
          { key: "confidence", label: "Confidence", render: (f) => confidence(f.confidence) },
          { key: "extractor", label: "Source", cls: "hide-sm", render: (f) => (f.extractor === "rules" ? "Rule" : fmt.title(f.extractor)) },
        ],
        rows: fieldRows,
        rowKey: (f) => f.name,
        onRowClick: (f) => focusSpan(f.name),
        emptyTitle: r.decision === "rejected" ? "Nothing was extracted" : "No fields",
        emptyText: r.decision === "rejected" ? "A quarantined document is never processed." : "",
      }),
    });
    const findingsCard = Kit.card({
      title: "Findings",
      subtitle: "Severity and blocking are separate: a critical lab value is preserved for the clinician, an injected instruction is not",
      flush: true,
      body: Kit.table({
        columns: [
          { key: "severity", label: "Severity", render: (i) => severityBadge(i.severity), sortValue: (i) => SEVERITY_RANK[i.severity] },
          { key: "message", label: "Finding", render: (i) => h("div", null, i.message, h("div", { class: "small muted" }, h("code", null, i.code), " · " + i.agent + (i.field ? " · " + i.field : ""))) },
          { key: "blocking", label: "Blocks", render: (i) => (i.blocking ? Kit.badge("Blocks", "critical") : h("span", { class: "muted small" }, "no")) },
        ],
        rows: findings,
        emptyTitle: "No findings",
        emptyText: "Every check passed.",
      }),
    });
    const pipelineCard = Kit.card({
      title: "Agent pipeline",
      subtitle: "Order is data, not control flow: each document type has its own route",
      body: h(
        "div",
        { class: "timeline" },
        r.pipeline.map((p) =>
          h(
            "div",
            { class: "tl-item" },
            h("span", { class: "tl-dot " + (p.ran ? (p.issues ? "warning" : "good") : "skip"), "aria-hidden": "true" }, p.ran ? (p.issues ? "!" : "✓") : ""),
            h(
              "div",
              null,
              h("div", { class: "row-between" }, h("b", null, fmt.title(p.agent)), p.ran ? h("span", { class: "small muted num" }, fmt.ms(p.elapsed_ms)) : h("span", { class: "small muted" }, "skipped")),
              h("div", { class: "small muted" }, p.ran ? AGENT_NOTE[p.agent] + " · confidence " + fmt.num(p.confidence, 2) + (p.fields ? " · " + p.fields + " fields" : "") + (p.issues ? " · " + p.issues + " issues" : "") : p.skipped_reason),
            ),
          ),
        ),
      ),
    });
    const summaryCard = r.summary
      ? Kit.card({
          title: "Summary for the reviewer",
          subtitle: "Built from verified fields only (" + (r.summary_source || "template") + "). Identifiers restored for the reviewer; the model never saw them.",
          body: h("div", { class: "codebox" }, r.summary),
        })
      : null;
    const policyCard = r.policy_context.length
      ? Kit.card({
          title: "Policy context",
          subtitle: "Passages the retriever cited for this document",
          flush: true,
          body: h(
            "div",
            null,
            r.policy_context.map((p) =>
              h(
                "div",
                { class: "passage" },
                h("div", { class: "row-between" }, h("b", null, p.title), h("code", { class: "small" }, p.citation)),
                h("div", { class: "ptext" }, Kit.highlight(p.text, termSpans(p.text, p.matched_terms))),
              ),
            ),
          ),
        })
      : null;
    host.append(
      h(
        "div",
        { class: "split mt" },
        h("div", { class: "stack" }, documentPanel(r), pipelineCard),
        h("div", { class: "stack" }, fieldsCard, findingsCard, summaryCard, policyCard),
      ),
    );
  }

  function termSpans(text, terms) {
    if (!terms || !terms.length) return [];
    const re = new RegExp("\\b(" + terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") + ")\\w*", "gi");
    const out = [];
    let m;
    while ((m = re.exec(text)) !== null) {
      out.push({ start: m.index, end: m.index + m[0].length, cls: "m-info" });
      if (!m[0].length) re.lastIndex++;
    }
    return out;
  }

  async function reviewView(view, params) {
    const all = await samples();
    const host = h("div", null);
    const picker = Kit.select([{ value: "", label: "Open a sample document…" }].concat(all.map((s) => ({ value: s.doc_id, label: s.doc_id + " · " + s.label }))), params.doc || "", (v) => v && app.go("review", { doc: v }), { style: { width: "320px", maxWidth: "100%" }, "aria-label": "Sample documents" });
    const text = h("textarea", { class: "textarea", rows: "8", placeholder: "Paste a referral, discharge summary, lab report or prior authorization…", "aria-label": "Document text" });
    const go = h("button", { class: "btn btn-primary", type: "submit" }, "Process document");
    const composer = h("form", { class: "stack" }, text, h("div", { class: "row-between" }, h("span", { class: "small muted" }, "Runs locally. Identifiers are minimised before any agent sees the text."), go));
    const process = (body, label) =>
      Kit.withButton(go, async () => {
        go.textContent = "Processing…";
        const r = await Kit.busy(host, () => api.post("/v1/ui/review", body));
        renderReview(host, r, label);
      });
    composer.addEventListener("submit", (e) => {
      e.preventDefault();
      if (text.value.trim()) process({ text: text.value }, "pasted document");
    });
    view.append(
      Kit.pageHead("Document review", "Every value on this page points at the text it came from. Nothing reaches a clinician without its evidence.", [picker]),
      h("details", { class: "card", style: { padding: "14px 18px" } }, h("summary", { style: { cursor: "pointer", fontWeight: "600" } }, "Process a new document"), h("div", { style: { marginTop: "12px" } }, composer)),
      h("div", { class: "mt" }, host),
    );
    const sample = all.find((s) => s.doc_id === params.doc) || (!params.doc && all[2]);
    if (sample) {
      text.value = sample.text;
      process({ text: sample.text, doc_id: sample.doc_id }, sample.label);
    }
  }

  // ------------------------------------------------------------------ privacy
  async function privacyView(view) {
    const all = await samples();
    const text = h("textarea", { class: "textarea", rows: "6", "aria-label": "Text to screen" });
    text.value = all.length ? all[0].text : "";
    const run = h("button", { class: "btn btn-primary", type: "submit" }, "Screen");
    const out = h("div", null);
    const chips = h("div", { class: "chips" }, all.map((s) => h("button", { class: "chip", type: "button", title: s.label, onclick: () => ((text.value = s.text), go()) }, s.doc_id + " · " + s.label)));
    const go = () =>
      Kit.withButton(run, async () => {
        const r = await Kit.busy(out, () => api.post("/v1/ui/screen", { text: text.value }));
        Kit.clear(out);
        out.append(
          r.would_be_rejected
            ? Kit.banner({ kind: "critical", title: "Would be quarantined", text: r.injection_issues.length + " instruction" + (r.injection_issues.length === 1 ? "" : "s") + " addressed to the model. The workflow would reject this document before any agent runs." })
            : Kit.banner({ kind: "good", title: r.findings.length + " identifier" + (r.findings.length === 1 ? "" : "s") + " minimised, no injected instructions", text: r.entities.length ? "Entities: " + r.entities.join(", ") + ". This is exactly what would leave the boundary." : "No identifiers found." }),
          h(
            "div",
            { class: "side-by-side mt" },
            Kit.card({ title: "Original", subtitle: "Identifiers highlighted", body: h("div", { class: "doc" }, Kit.highlight(r.original_text, r.findings.map((f) => ({ start: f.start, end: f.end, cls: "m-critical", title: f.entity })))) }),
            Kit.card({ title: "What a model would receive", subtitle: "Pseudonyms are stable within a run so agents can tell patients apart", body: h("div", { class: "doc" }, Kit.highlight(r.redacted_text, r.findings.map((f) => ({ start: f.redacted_start, end: f.redacted_end, cls: "m-good", title: f.entity })))) }),
          ),
          h(
            "div",
            { class: "grid cols-2 mt" },
            Kit.card({
              title: "Identifiers",
              flush: true,
              body: Kit.table({
                columns: [
                  { key: "entity", label: "Entity" },
                  { key: "replacement", label: "Replaced with", render: (f) => h("code", null, f.replacement) },
                  { key: "pseudonymised", label: "Treatment", render: (f) => (f.pseudonymised ? "Pseudonymised (agents need it)" : "Removed") },
                ],
                rows: r.findings,
                emptyTitle: "No identifiers found",
              }),
            }),
            Kit.card({
              title: "Injection screening",
              flush: true,
              body: Kit.table({
                columns: [
                  { key: "code", label: "Code", render: (i) => h("code", null, i.code) },
                  { key: "message", label: "Finding" },
                  { key: "blocking", label: "Blocks", render: (i) => (i.blocking ? Kit.badge("Blocks", "critical") : "no") },
                ],
                rows: r.injection_issues,
                emptyTitle: "No instructions addressed to a model",
              }),
            }),
          ),
        );
      });
    const form = h("form", { class: "stack" }, chips, text, h("div", { class: "row-between" }, h("span", { class: "small muted" }, "The pre-flight check a privacy reviewer asks for: nothing is processed or stored."), run));
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      go();
    });
    view.append(Kit.pageHead("Privacy screen", "Exactly what PHI minimisation and injection screening do to a document, before any agent or model sees it."), Kit.card({ body: form }), h("div", { class: "mt" }, out));
    go();
  }

  // ------------------------------------------------------------------ policies
  async function policiesView(view, params) {
    const input = h("input", { class: "input", type: "search", value: params.q || "anticoagulant drug interactions", "aria-label": "Search policies" });
    const run = h("button", { class: "btn btn-primary", type: "submit" }, "Search");
    const hitsHost = h("div", null);
    const first = await api.get("/v1/ui/kb");
    const go = () =>
      Kit.withButton(run, async () => {
        const r = await Kit.busy(hitsHost, () => api.get("/v1/ui/kb?q=" + encodeURIComponent(input.value)));
        Kit.clear(hitsHost);
        hitsHost.append(
          Kit.card({
            title: "Results",
            subtitle: "BM25 over policy and guideline passages, with the terms that matched",
            flush: true,
            body: r.hits.length
              ? h(
                  "div",
                  null,
                  r.hits.map((p) =>
                    h(
                      "div",
                      { class: "passage" },
                      h("div", { class: "row-between" }, h("div", null, h("b", null, p.title), h("span", { class: "muted small" }, " · " + p.section)), h("span", { class: "row" }, h("code", { class: "small" }, p.citation), h("span", { class: "small muted num" }, "score " + fmt.num(p.score, 2)))),
                      h("div", { class: "ptext" }, Kit.highlight(p.text, termSpans(p.text, p.matched_terms))),
                    ),
                  ),
                )
              : Kit.empty("No passage matched", "Try different wording."),
          }),
        );
      });
    const suggestions = ["lumbar MRI prior authorization", "anticoagulant drug interactions", "discharge summary required elements", "critical lab value notification", "renal dosing creatinine", "referral completeness"];
    const form = h(
      "form",
      { class: "stack" },
      h("div", { class: "row", style: { flexWrap: "nowrap" } }, input, run),
      h("div", { class: "chips" }, suggestions.map((q) => h("button", { class: "chip", type: "button", onclick: () => ((input.value = q), go()) }, q))),
    );
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      go();
    });
    view.append(
      Kit.pageHead("Policy library", first.passages + " passages from " + first.documents.length + " synthetic policies and guidelines. The retriever cites these in every summary."),
      h(
        "div",
        { class: "split-r" },
        h("div", { class: "stack" }, Kit.card({ body: form }), hitsHost),
        Kit.card({
          title: "Documents",
          flush: true,
          body: Kit.table({ columns: [{ key: "title", label: "Policy or guideline" }, { key: "sections", label: "Sections", align: "right" }], rows: first.documents, sort: { key: "title", dir: "asc" } }),
        }),
      ),
    );
    go();
  }

  // ------------------------------------------------------------------ audit
  async function auditView(view) {
    const data = await api.get("/v1/ui/audit");
    const detail = h("div", null, Kit.card({ body: Kit.empty("Select a run", "Its checksum, fields and agents appear here.") }));
    view.append(
      Kit.pageHead("Audit trail", fmt.int(data.total) + " runs recorded this session. Each record holds the decision, the fields with provenance and the agent trace.", [h("button", { class: "btn", type: "button", onclick: () => app.refresh() }, "Refresh")]),
      h(
        "div",
        { class: "split-r" },
        Kit.card({
          title: "Runs",
          subtitle: "Newest first",
          flush: true,
          body: Kit.table({
            columns: [
              { key: "doc_id", label: "Document", render: (r) => h("b", null, r.doc_id) },
              { key: "doc_type", label: "Type", cls: "hide-sm", render: (r) => docType(r.doc_type) },
              { key: "decision", label: "Decision", render: (r) => decisionBadge(r.decision) },
              { key: "max_severity", label: "Worst finding", cls: "hide-sm", render: (r) => severityBadge(r.max_severity) },
              { key: "mean_confidence", label: "Confidence", align: "right", format: (v) => fmt.num(v, 2) },
              { key: "duration_ms", label: "Time", align: "right", cls: "hide-sm", format: (v) => fmt.ms(v) },
            ],
            rows: data.records,
            maxHeight: 600,
            rowKey: (r) => r.run_id,
            emptyTitle: "No runs yet",
            emptyText: "Open the review queue or process a document.",
            onRowClick: (r) => {
              Kit.clear(detail);
              detail.append(
                Kit.card({
                  title: r.doc_id,
                  subtitle: "Run " + r.run_id,
                  body: h(
                    "div",
                    { class: "stack" },
                    Kit.kv([
                      ["Decision", decisionBadge(r.decision)],
                      ["Document type", docType(r.doc_type)],
                      ["Checksum", h("code", null, r.checksum)],
                      ["Agents", r.agents.map(fmt.title).join(" → ")],
                      ["Fields", r.fields.length ? r.fields.map(fieldLabel).join(", ") : "none"],
                    ]),
                    r.review_reasons.length ? h("div", null, h("div", { class: "small muted" }, "Review reasons"), h("ul", { style: { margin: "4px 0 0", paddingLeft: "18px" } }, r.review_reasons.map((x) => h("li", { class: "small" }, x)))) : null,
                  ),
                }),
              );
            },
          }),
        }),
        detail,
      ),
    );
  }

  const app = Kit.createApp({
    title: "Clinical Review Desk",
    health: "/health",
    healthLabel: (info) => "provider " + info.provider + " · " + info.knowledge_base_size + " policy passages",
    routes: [
      { path: "queue", label: "Review queue", render: queueView },
      { path: "review", label: "Document review", render: reviewView },
      { path: "privacy", label: "Privacy screen", render: privacyView },
      { path: "policies", label: "Policy library", render: policiesView },
      { path: "audit", label: "Audit trail", render: auditView },
    ],
  });
})();
