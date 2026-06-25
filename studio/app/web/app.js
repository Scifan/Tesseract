"use strict";

// Tesseract Studio front-end. Built on LiteGraph.js (a mature canvas node
// editor) for robust drag / connect / pan / zoom, with typed slots driven by
// the C++ block catalog. The browser is only a display surface — every ML op
// runs in the embedded C++ engine over the localhost JSON control plane.

const $ = (id) => document.getElementById(id);

const state = {
  catalog: {},          // kind -> spec
  graph: null,          // LGraph
  canvas: null,         // LGraphCanvas
  selected: null,       // selected LGraphNode
  poll: null,
  lossData: [],
  validateTimer: null,
};

// Per-category title-bar colors (LiteGraph node.color / bgcolor).
const CAT_COLOR = {
  Data:       ["#3a6ea5", "#1d2b3a"],
  Layers:     ["#4f8cff", "#1b2740"],
  Tensor:     ["#9b59b6", "#2a1d33"],
  Model:      ["#2bb673", "#16302a"],
  Loss:       ["#e07b53", "#33211a"],
  Optimizer:  ["#d9a441", "#332a16"],
  Tokenizer:  ["#56b6c2", "#16302f"],
  Train:      ["#c25b8c", "#331a28"],
  Inference:  ["#7d8cff", "#1f2140"],
};

// Map a catalog PortType to a LiteGraph slot type ("Any" -> 0 wildcard).
const slotType = (t) => (t === "Any" ? 0 : t);

// --------------------------------------------------------------------------- //
// Catalog -> LiteGraph node types
// --------------------------------------------------------------------------- //
function registerCatalog(specs) {
  // Reset (so re-loading the catalog never double-registers).
  LiteGraph.registered_node_types = {};
  LiteGraph.searchbox_extras = {};

  for (const spec of specs) {
    state.catalog[spec.kind] = spec;
    const kind = spec.kind;

    function NodeCtor() {
      this.tsbKind = kind;
      for (const p of spec.inputs) this.addInput(p.name, slotType(p.type));
      for (const p of spec.outputs) this.addOutput(p.name, slotType(p.type));
      this.properties = {};
      for (const ps of spec.params) {
        this.properties[ps.name] = ps.default;
        this.addParamWidget(ps);
      }
      const colors = CAT_COLOR[spec.category] || ["#666", "#222"];
      this.color = colors[0];
      this.bgcolor = colors[1];
      this.size = this.computeSize();
      if (this.size[0] < 168) this.size[0] = 168;
    }

    NodeCtor.title = spec.label;
    NodeCtor.desc = spec.summary;
    NodeCtor.prototype.addParamWidget = function (ps) {
      const self = this;
      const cb = (v) => { self.properties[ps.name] = v; onParamChanged(self); };
      if (ps.type === "bool") {
        this.addWidget("toggle", ps.name, !!ps.default, cb);
      } else if (ps.type === "enum") {
        this.addWidget("combo", ps.name, ps.default, cb, { values: ps.options || [] });
      } else if (ps.type === "int") {
        this.addWidget("number", ps.name, ps.default, cb, { precision: 0, step: 10 });
      } else if (ps.type === "float") {
        this.addWidget("number", ps.name, ps.default, cb, { precision: 4, step: 1 });
      } else {
        this.addWidget("text", ps.name, String(ps.default ?? ""), cb);
      }
    };

    // Draw inferred shape + diagnostic dot under the node.
    NodeCtor.prototype.onDrawForeground = function (ctx) {
      if (this.flags && this.flags.collapsed) return;
      if (this._shapeText) {
        ctx.save();
        ctx.font = "10px monospace";
        ctx.fillStyle = "#8fd0ff";
        ctx.textAlign = "right";
        ctx.fillText(this._shapeText, this.size[0] - 6, this.size[1] - 6);
        ctx.restore();
      }
      if (this._sev) {
        ctx.save();
        ctx.fillStyle = this._sev === "error" ? "#ff5d5d" : "#f7c948";
        ctx.beginPath();
        ctx.arc(this.size[0] - 8, -LiteGraph.NODE_TITLE_HEIGHT * 0.5, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    };

    NodeCtor.prototype.onSelected = function () { showInspector(this); };

    LiteGraph.registerNodeType("tesseract/" + kind, NodeCtor);
  }
}

// --------------------------------------------------------------------------- //
// Palette
// --------------------------------------------------------------------------- //
function buildPalette(filter) {
  const byCat = {};
  for (const k in state.catalog) {
    const s = state.catalog[k];
    if (filter && !(`${s.label} ${s.kind} ${s.summary}`.toLowerCase().includes(filter)))
      continue;
    (byCat[s.category] = byCat[s.category] || []).push(s);
  }
  const list = $("palette-list");
  list.innerHTML = "";
  for (const cat of Object.keys(byCat)) {
    const t = document.createElement("div");
    t.className = "cat-title";
    const dot = document.createElement("span");
    dot.className = "cat-dot";
    dot.style.background = (CAT_COLOR[cat] || ["#888"])[0];
    t.appendChild(dot);
    t.appendChild(document.createTextNode(cat));
    list.appendChild(t);
    for (const s of byCat[cat]) {
      const b = document.createElement("div");
      b.className = "palette-block";
      b.draggable = true;
      b.innerHTML = `<span class="pb-label">${s.label}</span><small>${s.summary}</small>`;
      b.onclick = () => addNodeCentered(s.kind);
      b.ondragstart = (e) => e.dataTransfer.setData("text/kind", s.kind);
      list.appendChild(b);
    }
  }
}

function addNode(kind, x, y) {
  const node = LiteGraph.createNode("tesseract/" + kind);
  if (!node) return null;
  node.pos = [x, y];
  state.graph.add(node);
  scheduleValidate();
  return node;
}

function addNodeCentered(kind) {
  const c = state.canvas;
  const r = c.canvas.getBoundingClientRect();
  const center = c.convertOffsetToCanvas
    ? c.convertOffsetToCanvas([r.width / 2, r.height / 2])
    : [200, 200];
  addNode(kind, center[0], center[1]);
}

// --------------------------------------------------------------------------- //
// Graph <-> .tsb
// --------------------------------------------------------------------------- //
function graphToTsb() {
  const nodes = [];
  for (const n of state.graph._nodes) {
    nodes.push({
      id: n.id, kind: n.tsbKind,
      params: Object.assign({}, n.properties),
      x: n.pos[0], y: n.pos[1],
    });
  }
  const edges = [];
  for (const id in state.graph.links) {
    const l = state.graph.links[id];
    if (!l) continue;
    const from = state.graph.getNodeById(l.origin_id);
    const to = state.graph.getNodeById(l.target_id);
    if (!from || !to) continue;
    const fp = from.outputs[l.origin_slot], tp = to.inputs[l.target_slot];
    if (!fp || !tp) continue;
    edges.push({ from: { node: from.id, port: fp.name }, to: { node: to.id, port: tp.name } });
  }
  return {
    name: $("graph-name").value || "untitled",
    device: $("device").value,
    nodes, edges,
  };
}

function outSlot(node, name) {
  for (let i = 0; i < node.outputs.length; i++) if (node.outputs[i].name === name) return i;
  return -1;
}
function inSlot(node, name) {
  for (let i = 0; i < node.inputs.length; i++) if (node.inputs[i].name === name) return i;
  return -1;
}

function applyParams(node, params) {
  if (!params) return;
  for (const k in params) {
    node.properties[k] = params[k];
    if (node.widgets) for (const w of node.widgets) if (w.name === k) w.value = params[k];
  }
}

function loadTsb(g) {
  state.graph.clear();
  state.selected = null;
  $("graph-name").value = g.name || "untitled";
  if (g.device) $("device").value = g.device;
  const idmap = {};
  for (const n of g.nodes || []) {
    const node = addNode(n.kind, n.x || 0, n.y || 0);
    if (!node) continue;
    applyParams(node, n.params);
    node.size = node.computeSize();
    if (node.size[0] < 168) node.size[0] = 168;
    idmap[n.id] = node;
  }
  for (const e of g.edges || []) {
    const a = idmap[e.from.node], b = idmap[e.to.node];
    if (!a || !b) continue;
    const os = outSlot(a, e.from.port), is = inSlot(b, e.to.port);
    if (os >= 0 && is >= 0) a.connect(os, b, is);
  }
  state.canvas.setDirty(true, true);
  showInspector(null);
  scheduleValidate();
}

// --------------------------------------------------------------------------- //
// Inspector
// --------------------------------------------------------------------------- //
function showInspector(node) {
  state.selected = node;
  const body = $("inspector-body");
  if (!node) {
    body.innerHTML = '<p class="hint">Select a block to edit its parameters.</p>';
    return;
  }
  const spec = state.catalog[node.tsbKind];
  body.innerHTML = `<p class="insp-title">${spec.label}</p><p class="hint">${spec.summary}</p>`;
  if (node._shapeText) {
    const sp = document.createElement("div");
    sp.className = "insp-shape";
    sp.textContent = "output " + node._shapeText;
    body.appendChild(sp);
  }
  for (const ps of spec.params) {
    const field = document.createElement("div");
    field.className = "field";
    const label = document.createElement("label");
    label.textContent = ps.label || ps.name;
    field.appendChild(label);
    let input;
    if (ps.type === "bool") {
      input = document.createElement("select");
      input.innerHTML = '<option value="true">true</option><option value="false">false</option>';
      input.value = String(node.properties[ps.name]);
      input.onchange = () => setParam(node, ps.name, input.value === "true");
    } else if (ps.type === "enum") {
      input = document.createElement("select");
      input.innerHTML = (ps.options || []).map((o) => `<option value="${o}">${o}</option>`).join("");
      input.value = node.properties[ps.name];
      input.onchange = () => setParam(node, ps.name, input.value);
    } else {
      input = document.createElement("input");
      input.type = (ps.type === "int" || ps.type === "float") ? "number" : "text";
      if (ps.type === "float") input.step = "any";
      input.value = node.properties[ps.name];
      input.onchange = () => {
        const v = ps.type === "int" ? parseInt(input.value)
          : ps.type === "float" ? parseFloat(input.value) : input.value;
        setParam(node, ps.name, v);
      };
    }
    field.appendChild(input);
    body.appendChild(field);
  }
  const del = document.createElement("button");
  del.className = "del-btn";
  del.textContent = "Delete block";
  del.onclick = () => { state.graph.remove(node); showInspector(null); scheduleValidate(); };
  body.appendChild(del);
}

function setParam(node, name, value) {
  node.properties[name] = value;
  if (node.widgets) for (const w of node.widgets) if (w.name === name) w.value = value;
  state.canvas.setDirty(true, true);
  scheduleValidate();
}

function onParamChanged(node) {
  if (state.selected === node) showInspector(node);
  scheduleValidate();
}

// --------------------------------------------------------------------------- //
// Validation (debounced)
// --------------------------------------------------------------------------- //
function scheduleValidate() {
  clearTimeout(state.validateTimer);
  state.validateTimer = setTimeout(validate, 180);
}

async function validate() {
  let res;
  try {
    res = await (await fetch("/api/validate", { method: "POST", body: JSON.stringify(graphToTsb()) })).json();
  } catch (e) { return; }

  for (const n of state.graph._nodes) { n._shapeText = null; n._sev = null; n.boxcolor = null; }
  if (res.shapes) {
    for (const id in res.shapes) {
      const n = state.graph.getNodeById(parseInt(id));
      if (n) n._shapeText = "[" + res.shapes[id].join(", ") + "]";
    }
  }
  for (const d of res.diagnostics || []) {
    const n = state.graph.getNodeById(d.node);
    if (n) {
      if (d.severity === "error" || !n._sev) n._sev = d.severity;
      n.boxcolor = d.severity === "error" ? "#ff5d5d" : "#f7c948";
    }
  }
  state.canvas.setDirty(true, true);

  const pane = $("pane-diag");
  if (!res.diagnostics || res.diagnostics.length === 0) {
    pane.innerHTML = '<div class="diag ok">No problems. Graph is ready to run.</div>';
  } else {
    pane.innerHTML = res.diagnostics.map((d) => {
      const where = d.node >= 0 ? `[node ${d.node}] ` : "";
      return `<div class="diag ${d.severity}">${d.severity.toUpperCase()}: ${where}${d.message}</div>`;
    }).join("");
  }
  if (state.selected) showInspector(state.selected);
  return res;
}

// --------------------------------------------------------------------------- //
// Run + events
// --------------------------------------------------------------------------- //
async function run() {
  const v = await validate();
  if (v && v.ok === false) { showTab("diag"); return; }
  state.lossData = [];
  drawChart();
  $("gen-text").textContent = "";
  $("gen-tokens").innerHTML = "";
  $("train-stats").textContent = "";
  $("log-out").innerHTML = "";
  $("tensor-grid").innerHTML = "";
  const r = await fetch("/api/run", { method: "POST", body: JSON.stringify(graphToTsb()) });
  if (!r.ok) { flashLog("run failed: " + await r.text()); return; }
  $("btn-run").disabled = true;
  $("btn-stop").disabled = false;
  pollEvents(0);
}

function pollEvents(since) {
  state.poll = setTimeout(async () => {
    let res;
    try { res = await (await fetch("/api/events?since=" + since)).json(); }
    catch (e) { return pollEvents(since); }
    for (const ev of res.events) handleEvent(ev);
    if (res.running) pollEvents(res.next);
    else { $("btn-run").disabled = false; $("btn-stop").disabled = true; }
  }, 150);
}

function handleEvent(ev) {
  const d = ev.data || {};
  if (ev.type === "loss") {
    state.lossData.push(d.loss);
    drawChart();
    $("train-stats").textContent =
      `step ${d.step} · epoch ${d.epoch} · loss ${d.loss.toFixed(4)} · acc ${(d.acc * 100).toFixed(1)}%`;
    showTab("train");
  } else if (ev.type === "metric") {
    flashLog(`epoch ${d.epoch}: loss ${d.loss.toFixed(4)}, acc ${(d.acc * 100).toFixed(1)}%`);
  } else if (ev.type === "token") {
    const t = document.createElement("span");
    t.className = "tok";
    t.textContent = d.text !== undefined ? d.text : ("#" + d.id);
    $("gen-tokens").appendChild(t);
    if (d.text !== undefined) $("gen-text").textContent = d.text;
    showTab("out");
  } else if (ev.type === "text") {
    $("gen-text").textContent = d.text;
  } else if (ev.type === "tensor") {
    renderTensor(d);
    showTab("tensors");
  } else if (ev.type === "log") {
    flashLog(d.message);
  } else if (ev.type === "error") {
    $("pane-diag").innerHTML += `<div class="diag error">ERROR: ${d.message}</div>`;
    flashLog("ERROR: " + d.message);
    showTab("diag");
  } else if (ev.type === "done") {
    flashLog("done.");
  }
}

async function stop() { await fetch("/api/stop", { method: "POST" }); }

// --------------------------------------------------------------------------- //
// Tensor heatmaps (Tensor / autograd playground)
// --------------------------------------------------------------------------- //
function renderTensor(d) {
  const grid = $("tensor-grid");
  if (grid.classList.contains("hint")) { grid.classList.remove("hint"); grid.innerHTML = ""; }
  const card = document.createElement("div");
  card.className = "tcard role-" + (d.role || "value");

  const head = document.createElement("div");
  head.className = "tcard-head";
  const role = d.role === "result" ? "output" : d.role === "grad" ? "gradient" : "value";
  head.innerHTML = `<b>${d.label || ("node " + d.node)}</b> <span class="tmeta">${role} · [${(d.shape || []).join(", ")}]</span>`;
  card.appendChild(head);

  const vals = d.values || [];
  let rows = 1, cols = vals.length;
  if (d.shape && d.shape.length >= 2) { rows = d.shape[0]; cols = Math.max(1, Math.round(vals.length / rows)); }
  const maxR = Math.min(rows, 24), maxC = Math.min(cols, 32);
  let lo = Infinity, hi = -Infinity;
  for (const v of vals) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;

  const hm = document.createElement("div");
  hm.className = "heatmap";
  hm.style.gridTemplateColumns = `repeat(${maxC}, 1fr)`;
  for (let i = 0; i < maxR; i++) {
    for (let j = 0; j < maxC; j++) {
      const v = vals[i * cols + j];
      const cell = document.createElement("div");
      cell.className = "cell";
      if (v !== undefined) {
        cell.style.background = diverging(v / m);
        cell.title = v.toFixed(4);
      } else cell.style.background = "transparent";
      hm.appendChild(cell);
    }
  }
  card.appendChild(hm);
  const foot = document.createElement("div");
  foot.className = "tcard-foot";
  foot.textContent = `min ${fmt(lo)} · max ${fmt(hi)}` +
    ((rows > maxR || cols > maxC) ? `  (showing ${maxR}×${maxC} of ${rows}×${cols})` : "");
  card.appendChild(foot);
  grid.appendChild(card);
}

function fmt(x) { return Number.isFinite(x) ? x.toFixed(3) : "—"; }

// Diverging blue→white→red colormap on t in [-1, 1].
function diverging(t) {
  t = Math.max(-1, Math.min(1, t));
  const mix = (a, b, k) => Math.round(a + (b - a) * k);
  if (t < 0) {
    const k = -t;
    return `rgb(${mix(245, 49, k)},${mix(247, 110, k)},${mix(250, 230, k)})`;
  }
  return `rgb(${mix(245, 230, t)},${mix(247, 80, t)},${mix(250, 70, t)})`;
}

// --------------------------------------------------------------------------- //
// IR + codegen
// --------------------------------------------------------------------------- //
async function showIr() {
  try {
    const r = await (await fetch("/api/ir", { method: "POST", body: JSON.stringify(graphToTsb()) })).json();
    $("ir-out").textContent = r.ir;
  } catch (e) { $("ir-out").textContent = "// IR unavailable: " + e; }
  showTab("ir");
}

async function codegen(lang) {
  const r = await (await fetch("/api/codegen?lang=" + lang, { method: "POST", body: JSON.stringify(graphToTsb()) })).json();
  $("code-out").textContent = r.code;
  showTab("code");
}

// --------------------------------------------------------------------------- //
// Save / open
// --------------------------------------------------------------------------- //
function save() {
  const g = graphToTsb();
  const blob = new Blob([JSON.stringify(g, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = (g.name || "graph") + ".tsb";
  a.click();
}

async function openFile(file) {
  const text = await file.text();
  const r = await fetch("/api/open", { method: "POST", body: text });
  if (!r.ok) { flashLog("open failed: " + await r.text()); return; }
  loadTsb(await r.json());
}

function flashLog(msg) {
  const line = document.createElement("div");
  line.className = "log-line";
  line.textContent = msg;
  $("log-out").appendChild(line);
  $("log-out").scrollTop = $("log-out").scrollHeight;
}

// --------------------------------------------------------------------------- //
// Loss chart
// --------------------------------------------------------------------------- //
function drawChart() {
  const c = $("loss-chart"), ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  const data = state.lossData;
  ctx.strokeStyle = "#26324d";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = (i / 4) * c.height;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(c.width, y); ctx.stroke();
  }
  if (data.length < 2) return;
  const max = Math.max(...data), min = Math.min(...data);
  const pad = 22, w = c.width - pad * 2, h = c.height - pad * 2;
  const grad = ctx.createLinearGradient(0, pad, 0, c.height - pad);
  grad.addColorStop(0, "#6fa8ff"); grad.addColorStop(1, "#2bb673");
  ctx.strokeStyle = grad;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  data.forEach((v, i) => {
    const x = pad + (i / (data.length - 1)) * w;
    const y = pad + (1 - (v - min) / (max - min || 1)) * h;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#8aa0c8";
  ctx.font = "11px monospace";
  ctx.fillText("loss " + max.toFixed(3), pad, pad - 6);
  ctx.fillText(min.toFixed(3), pad, c.height - 6);
}

// --------------------------------------------------------------------------- //
// Tabs
// --------------------------------------------------------------------------- //
function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".pane").forEach((p) => p.classList.toggle("active", p.id === "pane-" + name));
}

// --------------------------------------------------------------------------- //
// Canvas setup
// --------------------------------------------------------------------------- //
function setupCanvas() {
  const graph = new LGraph();
  const canvasEl = $("graph-canvas");
  const canvas = new LGraphCanvas(canvasEl, graph);
  canvas.background_image = null;
  canvas.render_canvas_border = false;
  canvas.links_render_mode = LiteGraph.SPLINE_LINK;
  canvas.always_render_background = false;
  canvas.clear_background_color = "#0e1422";
  canvas.node_title_color = "#dce6ff";
  canvas.default_link_color = "#5b7bd0";
  canvas.onNodeSelected = (n) => showInspector(n);
  canvas.onNodeDeselected = () => showInspector(null);
  // Re-validate when the user finishes connecting / removing wires.
  const origChange = graph.onConnectionChange;
  graph.onConnectionChange = function () {
    if (origChange) origChange.apply(this, arguments);
    scheduleValidate();
  };
  state.graph = graph;
  state.canvas = canvas;
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);

  const wrap = $("canvas-wrap");
  wrap.addEventListener("dragover", (e) => e.preventDefault());
  wrap.addEventListener("drop", (e) => {
    e.preventDefault();
    const kind = e.dataTransfer.getData("text/kind");
    if (!kind) return;
    const pos = canvas.convertEventToCanvasOffset(e);
    addNode(kind, pos[0], pos[1]);
  });
}

function resizeCanvas() {
  const wrap = $("canvas-wrap");
  const c = $("graph-canvas");
  c.width = wrap.clientWidth;
  c.height = wrap.clientHeight;
  if (state.canvas) state.canvas.resize(c.width, c.height);
}

// --------------------------------------------------------------------------- //
// Example graphs
// --------------------------------------------------------------------------- //
function exampleMlp() {
  loadTsb({
    name: "mlp_train", device: $("device").value,
    nodes: [
      { id: 1, kind: "Input", params: { batch: 32, features: 8 }, x: 40, y: 120 },
      { id: 2, kind: "Linear", params: { in_features: 8, out_features: 32, bias: true }, x: 250, y: 60 },
      { id: 3, kind: "ReLU", params: {}, x: 460, y: 60 },
      { id: 4, kind: "Linear", params: { in_features: 32, out_features: 3, bias: true }, x: 620, y: 60 },
      { id: 5, kind: "SequentialModel", params: {}, x: 840, y: 60 },
      { id: 6, kind: "SyntheticClassification", params: { samples: 256, features: 8, classes: 3, seed: 0 }, x: 250, y: 300 },
      { id: 7, kind: "Adam", params: { lr: 0.01 }, x: 840, y: 260 },
      { id: 8, kind: "CrossEntropyLoss", params: {}, x: 840, y: 400 },
      { id: 9, kind: "TrainLoop", params: { epochs: 40, batch_size: 32 }, x: 1090, y: 200 },
    ],
    edges: [
      { from: { node: 1, port: "out" }, to: { node: 2, port: "in" } },
      { from: { node: 2, port: "out" }, to: { node: 3, port: "in" } },
      { from: { node: 3, port: "out" }, to: { node: 4, port: "in" } },
      { from: { node: 4, port: "out" }, to: { node: 5, port: "in" } },
      { from: { node: 5, port: "model" }, to: { node: 7, port: "model" } },
      { from: { node: 5, port: "model" }, to: { node: 9, port: "model" } },
      { from: { node: 6, port: "data" }, to: { node: 9, port: "data" } },
      { from: { node: 7, port: "opt" }, to: { node: 9, port: "optimizer" } },
      { from: { node: 8, port: "loss" }, to: { node: 9, port: "loss" } },
    ],
  });
}

function exampleLlama() {
  loadTsb({
    name: "llama_generate", device: $("device").value,
    nodes: [
      { id: 1, kind: "LoadLlama", params: { model_dir: "", vocab_size: 256, hidden_size: 64, num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 4, intermediate_size: 128 }, x: 120, y: 120 },
      { id: 2, kind: "Generate", params: { prompt_ids: "1, 2, 3", max_new_tokens: 24 }, x: 460, y: 120 },
    ],
    edges: [{ from: { node: 1, port: "model" }, to: { node: 2, port: "model" } }],
  });
}

function exampleTensor() {
  loadTsb({
    name: "tensor_playground", device: $("device").value,
    nodes: [
      { id: 1, kind: "TensorConst", params: { rows: 3, cols: 4, init: "randn", scale: 1.0, seed: 1 }, x: 60, y: 60 },
      { id: 2, kind: "TensorConst", params: { rows: 4, cols: 2, init: "randn", scale: 1.0, seed: 2 }, x: 60, y: 280 },
      { id: 3, kind: "TMatMul", params: {}, x: 360, y: 150 },
      { id: 4, kind: "TReLU", params: {}, x: 580, y: 150 },
      { id: 5, kind: "TensorInspect", params: { backward: true }, x: 800, y: 150 },
    ],
    edges: [
      { from: { node: 1, port: "out" }, to: { node: 3, port: "a" } },
      { from: { node: 2, port: "out" }, to: { node: 3, port: "b" } },
      { from: { node: 3, port: "out" }, to: { node: 4, port: "in" } },
      { from: { node: 4, port: "out" }, to: { node: 5, port: "in" } },
    ],
  });
}

// --------------------------------------------------------------------------- //
// Wiring
// --------------------------------------------------------------------------- //
function wireUi() {
  $("btn-run").onclick = run;
  $("btn-stop").onclick = stop;
  $("btn-validate").onclick = validate;
  $("btn-ir").onclick = showIr;
  $("btn-cpp").onclick = () => codegen("cpp");
  $("btn-py").onclick = () => codegen("python");
  $("btn-save").onclick = save;
  $("btn-open").onclick = () => $("file-input").click();
  $("btn-clear").onclick = () => { state.graph.clear(); showInspector(null); scheduleValidate(); };
  $("file-input").onchange = (e) => { if (e.target.files[0]) openFile(e.target.files[0]); };
  $("device").onchange = () => scheduleValidate();
  $("graph-name").onchange = () => {};
  $("palette-search").oninput = (e) => buildPalette(e.target.value.trim().toLowerCase());
  document.querySelectorAll(".tab").forEach((t) => t.onclick = () => showTab(t.dataset.tab));

  const menu = $("examples-menu");
  $("btn-examples").onclick = (e) => { e.stopPropagation(); menu.classList.toggle("hidden"); };
  document.addEventListener("click", () => menu.classList.add("hidden"));
  menu.querySelectorAll("[data-ex]").forEach((el) => el.onclick = () => {
    menu.classList.add("hidden");
    if (el.dataset.ex === "mlp") exampleMlp();
    else if (el.dataset.ex === "llama") exampleLlama();
    else exampleTensor();
  });
}

(async function main() {
  setupCanvas();
  wireUi();
  const specs = await (await fetch("/api/catalog")).json();
  registerCatalog(specs);
  buildPalette("");
  exampleMlp();
})();
