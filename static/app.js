'use strict';
const $ = (id) => document.getElementById(id);
const form = $('analysis-form');
const sampleSelect = $('sample-select');
const uploadInput = $('upload-input');
const tabs = [...document.querySelectorAll('[role="tab"]')];
let samples = [];
let uploadFile = null;
let previewObjectURL = null;
let activeJob = null;
let pending = false;

function textElement(tag, text, className) {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}
function showError(message) {
  $('analysis-error').textContent = message;
  $('analysis-error').hidden = false;
}
function clearError() { $('analysis-error').hidden = true; }
function setStatus(message, state = 'running') {
  $('analysis-status').textContent = message;
  $('analysis-status').dataset.state = state;
  $('analysis-status').hidden = false;
}
function setBusy(value) {
  pending = value;
  $('input-fields').disabled = value || Boolean(activeJob);
  $('run-button').disabled = value;
  $('run-button').textContent = value ? 'Analysis in progress…' : (activeJob ? 'Reconnect to analysis →' : 'Run analysis →');
  $('results-panel').setAttribute('aria-busy', String(value));
}
async function request(url, options) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); }
  catch { throw new Error('The server returned an unreadable response. Check the server and try again.'); }
  if (!response.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : 'The request could not be completed. Check the image and settings.';
    throw new Error(detail);
  }
  return data;
}
function updateRange(input) {
  const output = document.querySelector(`output[for="${input.id}"]`);
  if (!output) return;
  output.textContent = (input.step === '0.01' ? Number(input.value).toFixed(2) : input.value) + (input.dataset.unit || '');
}
document.querySelectorAll('input[type="range"]').forEach((input) => {
  input.addEventListener('input', () => updateRange(input));
  updateRange(input);
});
function updatePreview() {
  if (previewObjectURL) { URL.revokeObjectURL(previewObjectURL); previewObjectURL = null; }
  const sample = samples.find((item) => item.name === sampleSelect.value);
  const url = uploadFile ? (previewObjectURL = URL.createObjectURL(uploadFile)) : sample?.url;
  const name = uploadFile?.name || sample?.name;
  $('preview-name').textContent = name || 'Choose an image to begin';
  $('input-preview').hidden = !url;
  $('preview-empty').hidden = Boolean(url);
  $('preview-zoom').disabled = !url;
  if (url) { $('input-preview').src = url; $('input-preview').alt = `Input preview: ${name}`; }
  else $('input-preview').removeAttribute('src');
  $('upload-info').hidden = !uploadFile;
  $('upload-name').textContent = uploadFile?.name || '';
}
$('input-preview').addEventListener('error', () => {
  $('input-preview').hidden = true;
  $('preview-empty').hidden = false;
  $('preview-empty').textContent = 'Preview unavailable. Please choose a valid image.';
  $('preview-zoom').disabled = true;
});
function chooseUpload(file) {
  if (pending || activeJob) return;
  clearError();
  if (file && !/\.(jpe?g|png|bmp|tiff?)$/i.test(file.name)) {
    showError('Unsupported image format. Please choose a JPG, PNG, BMP or TIFF image.');
    uploadInput.value = '';
    return;
  }
  if (file && file.size > 50 * 1024 * 1024) {
    showError('Image exceeds the 50 MB upload limit.');
    uploadInput.value = '';
    return;
  }
  uploadFile = file || null;
  updatePreview();
}
uploadInput.addEventListener('change', () => chooseUpload(uploadInput.files[0]));
sampleSelect.addEventListener('change', updatePreview);
$('clear-upload').addEventListener('click', () => { uploadInput.value = ''; chooseUpload(null); });
['dragenter', 'dragover'].forEach((name) => $('drop-zone').addEventListener(name, (event) => {
  event.preventDefault();
  if (!pending && !activeJob) $('drop-zone').classList.add('dragging');
}));
['dragleave', 'drop'].forEach((name) => $('drop-zone').addEventListener(name, (event) => {
  event.preventDefault(); $('drop-zone').classList.remove('dragging');
  if (name === 'drop' && event.dataTransfer.files.length) chooseUpload(event.dataTransfer.files[0]);
}));
$('reset-button').addEventListener('click', () => {
  if (pending || activeJob) return;
  form.reset(); sampleSelect.value = samples[0]?.name || ''; uploadFile = null;
  document.querySelectorAll('input[type="range"]').forEach(updateRange);
  $('result-content').hidden = true;
  $('idle-view').hidden = false;
  $('analysis-status').hidden = true;
  $('result-label').textContent = 'Ready when you are';
  clearError(); updatePreview();
});

function openImage(url, caption) {
  $('zoom-image').src = url;
  $('zoom-image').alt = caption;
  $('zoom-caption').textContent = caption;
  $('image-dialog').showModal();
  document.querySelector('.zoom-scroll').scrollTo(0, 0);
}
$('preview-zoom').addEventListener('click', () => openImage($('input-preview').src, $('input-preview').alt));
$('close-dialog').addEventListener('click', () => $('image-dialog').close());
$('image-dialog').addEventListener('click', (event) => {
  if (event.target !== $('image-dialog')) return;
  const rect = $('image-dialog').getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) $('image-dialog').close();
});
function selectTab(name, focus = false) {
  tabs.forEach((tab) => {
    const selected = tab.dataset.tab === name;
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
    $(tab.getAttribute('aria-controls')).hidden = !selected;
    if (selected && focus) tab.focus();
  });
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => selectTab(tab.dataset.tab));
  tab.addEventListener('keydown', (event) => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    if (next !== undefined) { event.preventDefault(); selectTab(tabs[next].dataset.tab, true); }
  });
});
$('comparison-columns').addEventListener('change', () => {
  $('comparison-images').style.setProperty('--comparison-cols', $('comparison-columns').value);
});
function renderImages(id, items) {
  const container = $(id);
  container.replaceChildren();
  if (!items.length) { container.append(textElement('p', 'No fruit available for this view.', 'empty-gallery')); return; }
  items.forEach((item) => {
    const figure = document.createElement('figure'); figure.className = 'image-card';
    const button = document.createElement('button'); button.type = 'button';
    button.setAttribute('aria-label', `Enlarge: ${item.caption}`);
    const image = document.createElement('img'); image.src = item.url; image.alt = item.caption; image.loading = 'lazy';
    button.append(image); button.addEventListener('click', () => openImage(item.url, item.caption));
    figure.append(button, textElement('figcaption', item.caption)); container.append(figure);
  });
}
function formatValue(value) {
  if (value === null || value === undefined) return '—';
  return typeof value === 'number' ? value.toLocaleString('en-US', { maximumFractionDigits: 6, useGrouping: false }) : String(value);
}
function renderResult(result) {
  const summary = result.summary;
  $('metrics').replaceChildren();
  const metrics = [
    ['Fruit detected', summary.fruit_count, 'individual specimens'],
    ['Processing time', summary.elapsed_seconds.toFixed(2), 'seconds · inference & measurements'],
    ['Image dimensions', `${summary.width} × ${summary.height}`, 'pixels'],
    ['Calibration', formatValue(summary.scale_ratio), `mm / pixel · ${summary.scale_source.replaceAll('_', ' ')}`],
  ];
  metrics.forEach(([label, value, unit]) => {
    const card = document.createElement('div'); card.className = 'metric';
    card.append(textElement('span', label, 'metric-label'), textElement('span', String(value), 'metric-value'), textElement('span', unit, 'metric-unit'));
    $('metrics').append(card);
  });
  $('warnings').replaceChildren(...result.warnings.map((message) => textElement('p', message, 'notice warning')));
  $('result-source').textContent = `${summary.filename} · ${summary.source === 'upload' ? 'Uploaded image' : 'Dataset sample'}`;
  $('download-zip').href = result.downloads.zip;
  $('download-csv').href = result.downloads.csv;
  $('download-manifest').href = result.downloads.manifest;
  renderImages('input-images', [result.images.original, result.images.roi_guide, result.images.detection_roi]);
  renderImages('detection-images', [result.images.threshold_mask, result.images.detection]);
  renderImages('crop-images', result.galleries.crops);
  renderImages('segmentation-images', result.galleries.segmentation);
  renderImages('mask-images', result.galleries.class_masks);
  renderImages('final-image', [result.images.full_segmentation]);
  renderImages('comparison-images', result.galleries.comparison);
  document.querySelectorAll('[data-legend]').forEach((container) => {
    container.replaceChildren();
    const entries = container.dataset.legend === 'basic' ? result.legend.slice(0, 5) : result.legend;
    entries.forEach((item) => {
      const entry = textElement('span', '', 'legend-item');
      const swatch = textElement('span', '', 'legend-swatch'); swatch.style.backgroundColor = `rgb(${item.color.join(',')})`;
      entry.append(swatch, document.createTextNode(item.label)); container.append(entry);
    });
  });
  const header = document.createElement('tr');
  result.features.columns.forEach((label) => { const cell = textElement('th', label); cell.scope = 'col'; header.append(cell); });
  $('feature-table').querySelector('thead').replaceChildren(header);
  const body = $('feature-table').querySelector('tbody'); body.replaceChildren();
  result.features.rows.forEach((row) => {
    const tr = document.createElement('tr');
    row.forEach((value) => { const td = textElement('td', formatValue(value)); if (value !== null) td.title = String(value); tr.append(td); });
    body.append(tr);
  });
  if (!result.features.rows.length) {
    const tr = document.createElement('tr'); const td = textElement('td', 'No fruit detected. The CSV includes the complete column headers.');
    td.colSpan = result.features.columns.length; tr.append(td); body.append(tr);
  }
  $('feature-count').textContent = `${result.features.rows.length} fruit · ${result.features.columns.length} columns`;
  $('run-config').replaceChildren();
  Object.entries({ Device: summary.device, ...summary.configuration }).forEach(([key, value]) => {
    $('run-config').append(textElement('dt', key.replaceAll('_', ' ')), textElement('dd', formatValue(value)));
  });
  selectTab('final');
  $('idle-view').hidden = true;
  $('result-content').hidden = false;
  $('result-label').textContent = 'Analysis complete';
}
async function pollJob() {
  try {
    for (;;) {
      const job = await request(activeJob.status_url);
      if (job.status === 'completed') {
        renderResult(job.result);
        setStatus('Analysis complete. Explore the results below.', 'completed');
        activeJob = null; setBusy(false); return;
      }
      if (job.status === 'failed') {
        activeJob = null; setBusy(false); $('analysis-status').hidden = true;
        $('result-label').textContent = 'Analysis failed'; showError(job.error); return;
      }
      setStatus(job.status === 'queued' ? 'Queued. Waiting for the current analysis to finish…' : 'Analyzing your specimen. First-time model loading and measurements may take a few minutes.');
      await new Promise((resolve) => setTimeout(resolve, 1200));
    }
  } catch (error) {
    $('analysis-status').hidden = true;
    showError(`${error.message} Your analysis may still be running. Use “Reconnect to analysis” to check the same run.`);
    setBusy(false);
  }
}
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (pending) return;
  clearError();
  if (activeJob) { setBusy(true); await pollJob(); return; }
  if (!uploadFile && !sampleSelect.value) { showError('Upload an image or choose a sample first.'); return; }
  const options = {};
  form.querySelectorAll('[name]').forEach((input) => {
    if (input.name !== 'sample_name') options[input.name] = input.tagName === 'SELECT' ? input.value : Number(input.value);
  });
  const payload = new FormData();
  payload.append('options', JSON.stringify(options));
  if (sampleSelect.value) payload.append('sample_name', sampleSelect.value);
  if (uploadFile) payload.append('upload', uploadFile);
  setBusy(true);
  $('result-content').hidden = true; $('idle-view').hidden = false;
  $('result-label').textContent = 'Preparing analysis';
  setStatus('Validating the input image and configuration…');
  try { activeJob = await request('/api/jobs', { method: 'POST', body: payload }); }
  catch (error) { $('analysis-status').hidden = true; $('result-label').textContent = 'Ready when you are'; showError(error.message); setBusy(false); return; }
  await pollJob();
});
async function initialize() {
  try {
    const data = await request('/api/samples'); samples = data.samples;
    sampleSelect.replaceChildren(new Option('Choose a dataset image', ''));
    samples.forEach((sample) => sampleSelect.add(new Option(sample.name, sample.name)));
    if (samples.length) {
      sampleSelect.value = samples[0].name;
      const hero = $('hero-image');
      hero.addEventListener('error', () => { hero.hidden = true; $('hero-placeholder').hidden = false; });
      hero.src = samples[0].url; hero.hidden = false; $('hero-placeholder').hidden = true;
    }
    updatePreview();
  } catch (error) {
    sampleSelect.replaceChildren(new Option('Samples unavailable · upload an image', ''));
    showError(error.message);
  }
}
initialize();
