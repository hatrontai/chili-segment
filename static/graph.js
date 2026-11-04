/* Live SSE and recorded-event replay. No model or inference calls originate here. */
(function () {
  'use strict';
  const byId = id => document.getElementById(id);
  const { nodes, fruitNodes, project, nodeState, interval } = window.PipelineTrace;
  const history = [];
  let job = null, stream = null, timer = null, finished = false, mode = 'live';
  let cursor = 0, follow = true, selectedNode = 'input', selectedCandidate = null;
  let finalResultsVisible = false, syncing = false;
  const labels = { pending: 'Waiting', running: 'Running', completed: 'Complete', failed: 'Failed', skipped: 'Skipped' };
  const icons = { pending: '○', running: '◉', completed: '✓', failed: '!', skipped: '–' };
  const buttons = new Map();
  const canvas = byId('pipeline-canvas');
  const svg = byId('pipeline-edges');
  const ns = 'http://www.w3.org/2000/svg';
  let outputKey = '';

  for (const [index, node] of nodes.entries()) {
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'pipeline-node'; button.dataset.node = node.id;
    button.setAttribute('aria-controls', 'node-output');
    button.append(textElement('span', String(index + 1).padStart(2, '0'), 'node-number'),
      textElement('strong', node.label), textElement('span', node.description, 'node-description'),
      textElement('span', '○ Waiting', 'node-status'));
    button.addEventListener('click', () => { selectedNode = node.id; follow = false; render(); });
    canvas.append(button); buttons.set(node.id, button);
  }

  function pause() { clearTimeout(timer); timer = null; }
  function restoreResults() {
    if (!finished) return;
    byId('result-content').hidden = !finalResultsVisible || (mode === 'replay' && cursor < history.length);
  }
  function view() { return project(history, mode === 'replay' ? cursor : history.length); }
  function renderOutputs(state, candidate) {
    const key = `${mode}:${selectedNode}:${candidate}:${state.seq || 0}`;
    if (key === outputKey) return;
    outputKey = key;
    const heading = nodes.find(n => n.id === selectedNode);
    byId('node-output-title').textContent = heading.label;
    byId('node-output-context').textContent = fruitNodes.has(selectedNode) && candidate !== null ? `Candidate ${candidate + 1}` : 'Whole image';
    byId('node-output-state').textContent = labels[state.state];
    byId('node-output-state').dataset.state = state.state;
    const outputs = state.outputs || {};
    const pictures = outputs.images || [];
    byId('node-output-images').hidden = !pictures.length;
    if (pictures.length) renderImages('node-output-images', pictures);
    else byId('node-output-images').replaceChildren();
    byId('node-output-details').replaceChildren();
    for (const [key, value] of Object.entries(outputs.details || {})) {
      byId('node-output-details').append(textElement('dt', key.replaceAll('_', ' ')),
        textElement('dd', value === null ? '—' : (typeof value === 'object' ? JSON.stringify(value) : String(value))));
    }
    const downloads = byId('node-output-downloads'); downloads.replaceChildren();
    for (const [name, url] of Object.entries(outputs.downloads || {})) {
      const a = textElement('a', `Download ${name.toUpperCase()} ↓`, 'button secondary');
      a.href = url; a.download = ''; downloads.append(a);
    }
    const tableRegion = byId('node-feature-region');
    tableRegion.hidden = !outputs.features;
    const table = byId('node-feature-table'); table.replaceChildren();
    if (outputs.features) {
      const head = document.createElement('thead'), header = document.createElement('tr');
      outputs.features.columns.forEach(label => { const th = textElement('th', label); th.scope = 'col'; header.append(th); });
      head.append(header); table.append(head);
      const body = document.createElement('tbody');
      outputs.features.rows.forEach(row => {
        const tr = document.createElement('tr'); row.forEach(value => tr.append(textElement('td', formatValue(value)))); body.append(tr);
      });
      if (!outputs.features.rows.length) {
        const tr = document.createElement('tr'), td = textElement('td', 'No accepted fruit. The complete column schema is retained.');
        td.colSpan = outputs.features.columns.length; tr.append(td); body.append(tr);
      }
      table.append(body);
    }
    const hasOutput = pictures.length || Object.keys(outputs.details || {}).length || outputs.features || downloads.children.length;
    const message = byId('node-output-message');
    message.hidden = Boolean(hasOutput) && !state.message;
    message.textContent = state.message || (state.state === 'running' ? 'This step is running. Its output will appear as soon as it is ready.' :
      state.state === 'pending' ? 'This step has not been reached yet.' : state.state === 'skipped' ? 'This step was skipped.' : 'Step completed.');
    message.className = state.state === 'failed' ? 'notice error' : 'node-output-message';
  }
  function render() {
    const projection = view();
    const candidate = selectedCandidate ?? projection.currentCandidate;
    for (const node of nodes) {
      const state = nodeState(projection, node.id, candidate);
      const button = buttons.get(node.id);
      button.dataset.state = state.state;
      button.setAttribute('aria-pressed', String(node.id === selectedNode));
      button.querySelector('.node-status').textContent = `${icons[state.state]} ${labels[state.state]}`;
      button.setAttribute('aria-label', `${node.label}: ${labels[state.state]}${fruitNodes.has(node.id) && candidate !== null ? `, candidate ${candidate + 1}` : ''}`);
    }
    const select = byId('graph-candidate');
    const options = [new Option('Latest candidate', '')];
    const count = projection.total ?? projection.candidates.size;
    for (let i = 0; i < count; i++) {
      const entry = projection.candidates.get(i);
      const suffix = entry?.state === 'rejected' ? ' · rejected' : entry?.state === 'failed' ? ' · failed' : entry?.state === 'completed' ? ` · Chili ${entry.fruit_id}` : entry ? ' · processing' : ' · waiting';
      options.push(new Option(`Candidate ${i + 1}${suffix}`, String(i)));
    }
    select.replaceChildren(...options);
    select.value = selectedCandidate === null ? '' : String(selectedCandidate);
    if (select.selectedIndex < 0) { selectedCandidate = null; select.value = ''; }
    select.disabled = !count;
    byId('graph-progress-count').textContent = projection.total === null ? (projection.terminal === 'failed' ? 'Stopped before detection' : 'Waiting for detection') : `${projection.processed} / ${projection.total} candidates processed`;
    byId('graph-live-note').textContent = !job ? 'Choose an image below and run analysis to bring this graph to life.' :
      mode === 'replay' ? 'Presentation replay · recorded events, condensed pauses' : finished ? (projection.terminal === 'failed' ? 'Run stopped by an error · completed outputs remain available' : 'Recorded run complete · click any node to inspect its output') :
      `${projection.currentCandidate === null ? 'Processing image' : `Processing candidate ${projection.currentCandidate + 1}`} · outputs appear as steps finish`;
    byId('follow-live').setAttribute('aria-pressed', String(follow && mode === 'live'));
    byId('follow-live').textContent = finished ? 'Show recorded run' : 'Follow live';
    byId('replay-button').disabled = !finished || !history.length || timer !== null;
    byId('replay-button').textContent = mode === 'replay' && cursor > 0 && cursor < history.length ? 'Resume' : 'Replay';
    byId('pause-replay').disabled = timer === null;
    byId('restart-replay').disabled = !finished || !history.length;
    byId('replay-speed').disabled = !finished;
    byId('replay-seek').disabled = !finished;
    byId('replay-seek').max = String(history.length);
    byId('replay-seek').value = String(mode === 'replay' ? cursor : history.length);
    byId('trace-time').textContent = `${(projection.elapsed || 0).toFixed(2)} s recorded · ${mode === 'live' ? history.length : cursor} / ${history.length} events`;
    byId('trace-mode').textContent = mode === 'replay' ? 'REPLAY' : finished ? 'RECORDED' : 'LIVE EXECUTION';
    renderOutputs(nodeState(projection, selectedNode, candidate), candidate);
    restoreResults();
    requestAnimationFrame(drawEdges);
  }

  function drawEdges() {
    const compact = window.matchMedia('(max-width: 640px)').matches;
    const columns = compact ? 2 : 5;
    nodes.forEach((node, index) => {
      const row = Math.floor(index / columns), column = index % columns;
      const button = buttons.get(node.id);
      button.style.gridRow = row + 1;
      button.style.gridColumn = row % 2 ? columns - column : column + 1;
    });
    const bounds = canvas.getBoundingClientRect();
    if (!bounds.width) return;
    svg.setAttribute('viewBox', `0 0 ${bounds.width} ${bounds.height}`);
    svg.replaceChildren();
    const defs = document.createElementNS(ns, 'defs');
    const marker = document.createElementNS(ns, 'marker');
    for (const [k, v] of Object.entries({id:'pipeline-arrow',viewBox:'0 0 10 10',refX:'9',refY:'5',markerWidth:'5',markerHeight:'5',orient:'auto-start-reverse'})) marker.setAttribute(k, v);
    const arrow = document.createElementNS(ns, 'path'); arrow.setAttribute('d', 'M 0 0 L 10 5 L 0 10 z'); arrow.setAttribute('fill', '#78917a');
    marker.append(arrow); defs.append(marker); svg.append(defs);
    const rect = id => {
      const r = buttons.get(id).getBoundingClientRect();
      return {x:r.left-bounds.left,y:r.top-bounds.top,w:r.width,h:r.height,cx:r.left-bounds.left+r.width/2,cy:r.top-bounds.top+r.height/2};
    };
    function line(from, to, type, label) {
      const a = rect(from), b = rect(to), path = document.createElementNS(ns, 'path');
      let d;
      if (type === 'flow') {
        if (Math.abs(a.cy-b.cy)<3) d = a.cx<b.cx ? `M ${a.x+a.w} ${a.cy} H ${b.x-3}` : `M ${a.x} ${a.cy} H ${b.x+b.w+3}`;
        else d = `M ${a.cx} ${a.y+a.h} V ${b.y-3}`;
      } else if (type === 'loop') {
        const rail=bounds.width-4;
        d=`M ${a.cx} ${a.y+a.h} v 17 H ${rail} V ${b.y-20} H ${b.cx} V ${b.y-3}`;
      } else if (!compact && from==='scale') {
        d=`M ${a.cx} ${a.y+a.h} V ${b.y-3}`;
      } else {
        const rail=from==='input'?5:bounds.width-12;
        d=`M ${a.cx} ${a.y+a.h} v 11 H ${rail} V ${b.y-12} H ${b.cx} V ${b.y-3}`;
      }
      path.setAttribute('d',d); path.setAttribute('class',`pipeline-edge ${type}`);
      path.dataset.from=from;path.dataset.to=to;
      path.setAttribute('marker-end','url(#pipeline-arrow)');
      if (type==='flow') path.dataset.state=buttons.get(from).dataset.state;
      const title=document.createElementNS(ns,'title');title.textContent=label || `${from} to ${to}`;path.append(title);svg.append(path);
    }
    nodes.slice(0,-1).forEach((node,i)=>line(node.id,nodes[i+1].id,'flow'));
    line('scale','traits','dependency','Scale calibration supplies physical measurement units');
    line('input','results','dependency','Original image supplies the comparison view');
    line('masks','results','dependency','Masks supply the complete image overlay');
    line('traits','crop','loop','Repeat crop, segmentation, masks and traits for the next candidate');
  }
  function followEvent(event) {
    if (!follow) return;
    if (event.node && event.state !== 'skipped') selectedNode=event.node;
    selectedCandidate=null;
  }
  function accept(event) {
    if (!Number.isInteger(event.seq) || event.seq <= (history.at(-1)?.seq || 0)) return;
    history.push(event);
    if (mode==='live') { cursor=history.length;followEvent(event);render(); }
  }
  async function sync() {
    if (!job || syncing) return;
    syncing=true;
    const run=job;
    try {
      const trace=await request(run.trace_url || `${run.status_url}/trace`);
      if (job!==run) return;
      // Merge in sequence even if an SSE event arrived while the request was in flight.
      const merged=new Map(history.map(e=>[e.seq,e]));
      trace.events.forEach(e=>merged.set(e.seq,e));
      const last=history.at(-1)?.seq || 0;
      history.splice(0,history.length,...[...merged.values()].sort((a,b)=>a.seq-b.seq));
      if(mode==='live') { cursor=history.length; if((history.at(-1)?.seq || 0)>last && follow) { selectedNode=project(history).latestNode;selectedCandidate=null; }render(); }
    } catch { byId('graph-connection').textContent='Trace connection interrupted'; }
    finally { syncing=false; }
  }
  function reset() {
    pause();if(stream)stream.close();stream=null;
    history.length=0;job=null;finished=false;mode='live';cursor=0;follow=true;
    selectedNode='input';selectedCandidate=null;finalResultsVisible=false;outputKey='';
    byId('graph-connection').textContent='Ready';render();
  }
  function start(run) {
    reset();job=run;render();
    if(!window.EventSource) { byId('graph-connection').textContent='Using status updates';return; }
    stream=new EventSource(run.events_url || `${run.status_url}/events`);
    stream.onopen=()=>{byId('graph-connection').textContent='Connected · live events';};
    stream.addEventListener('progress',message=>{
      if(job!==run)return;
      try { accept(JSON.parse(message.data)); }
      catch {byId('graph-connection').textContent='Trace event unavailable';}
    });
    stream.onerror=()=>{byId('graph-connection').textContent=finished?'Recorded run complete':'Reconnecting to live events…';};
  }
  async function finish() {
    await sync();
    if(stream)stream.close();stream=null;finished=true;
    finalResultsVisible=!byId('result-content').hidden;
    byId('graph-connection').textContent='History saved for this session';render();
  }
  function schedule() {
    if(cursor>=history.length) { pause();render();return; }
    timer=setTimeout(()=>{
      timer=null;const event=history[cursor++];followEvent(event);render();schedule();
    },interval(history,cursor,Number(byId('replay-speed').value)));
    render();
  }
  function play(restart=false) {
    if(!finished || !history.length)return;
    pause();
    if(restart || mode==='live' || cursor>=history.length) { cursor=0;selectedNode='input'; }
    mode='replay';follow=true;selectedCandidate=null;render();schedule();
  }
  byId('follow-live').addEventListener('click',()=>{
    pause();mode='live';cursor=history.length;follow=true;selectedCandidate=null;
    selectedNode=view().latestNode;render();
  });
  byId('graph-candidate').addEventListener('change',()=>{
    selectedCandidate=byId('graph-candidate').value===''?null:Number(byId('graph-candidate').value);
    follow=false;render();
  });
  byId('replay-button').addEventListener('click',()=>play());
  byId('pause-replay').addEventListener('click',()=>{pause();render();});
  byId('restart-replay').addEventListener('click',()=>play(true));
  byId('replay-seek').addEventListener('input',()=>{
    pause();mode='replay';cursor=Number(byId('replay-seek').value);selectedCandidate=null;
    if(follow)selectedNode=view().latestNode;render();
  });
  byId('replay-speed').addEventListener('change',()=>{if(timer!==null){pause();schedule();}});
  new ResizeObserver(drawEdges).observe(canvas);
  window.pipelineGraph={start,finish,reset,sync};
  reset();
})();
