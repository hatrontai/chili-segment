/* Pure trace projection shared by live rendering, replay and regression checks. */
(function (root) {
  'use strict';
  const nodes = [
    ['input', 'Input & ROI', 'Image and detection region'],
    ['models', 'Model Setup', 'Load or reuse model weights'],
    ['scale', 'Scale Calibration', 'QR scale or dataset estimate'],
    ['detection', 'Detection', 'Locate candidate fruit'],
    ['crop', 'Padded Crop', 'Isolate one candidate'],
    ['segmentation', 'Segmentation & Orientation', 'Predict, orient and re-predict if needed'],
    ['masks', 'Anatomy Masks', 'Tissues and whole-fruit threshold mask'],
    ['traits', 'Trait Extraction', 'Geometry, seeds, shape and color'],
    ['results', 'Results & Comparison', 'Combine accepted fruit'],
    ['export', 'Export', 'CSV, manifest and ZIP'],
  ].map(([id, label, description]) => ({ id, label, description }));
  const fruitNodes = new Set(['crop', 'segmentation', 'masks', 'traits']);
  function project(events, cursor = events.length) {
    const states = new Map();
    const candidates = new Map();
    let currentCandidate = null, total = null, latestNode = 'input', terminal = null;
    for (const event of events.slice(0, cursor)) {
      if (event.candidate_total !== undefined) total = event.candidate_total;
      if (event.candidate_index !== undefined) {
        currentCandidate = event.candidate_index;
        const entry = candidates.get(currentCandidate) || { state: 'running' };
        if (event.kind === 'candidate') Object.assign(entry, event);
        candidates.set(currentCandidate, entry);
      }
      if (event.node) {
        const scope = event.candidate_index ?? 'global';
        // Preserve completed output if a later diagnostic changes the state.
        const previous = states.get(`${scope}:${event.node}`);
        states.set(`${scope}:${event.node}`, { ...previous, ...event });
        if (event.state !== 'skipped') latestNode = event.node;
      }
      if (event.kind === 'job') terminal = event.state;
    }
    return { states, candidates, currentCandidate, total, latestNode, terminal,
      processed: [...candidates.values()].filter(c => ['completed', 'rejected'].includes(c.state)).length,
      elapsed: cursor ? events[Math.min(cursor, events.length) - 1].elapsed_seconds : 0 };
  }
  function nodeState(projection, node, candidate) {
    const scope = fruitNodes.has(node) && candidate !== null ? candidate : 'global';
    return projection.states.get(`${scope}:${node}`) ||
      (fruitNodes.has(node) ? projection.states.get(`global:${node}`) : null) || { node, state: 'pending' };
  }
  function interval(events, cursor, speed = 1) {
    if (cursor >= events.length) return null;
    const previous = cursor ? events[cursor - 1].elapsed_seconds : 0;
    return Math.min(2, Math.max(.6, events[cursor].elapsed_seconds - previous)) * 1000 / speed;
  }
  const api = { nodes, fruitNodes, project, nodeState, interval };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.PipelineTrace = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
