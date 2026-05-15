const titles = {
  home: ['Home', 'Manual task and background worker status.'],
  manual: ['Manual Task', 'Run fixed local worker tasks without debug parameters.'],
  worker: ['Automatic Task', 'Start, stop, and inspect the automatic worker.'],
  logs: ['Logs', 'Stage-level task output and summaries.'],
  settings: ['Settings', 'Local runtime settings.']
};

const logOutput = document.querySelector('#log-output');
let currentSettings = null;
let durationTimer = null;
let lastManualStatus = null;
let lastWorkerStatus = null;

document.querySelectorAll('.nav').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('.nav').forEach((item) => item.classList.remove('active'));
    document.querySelectorAll('.view').forEach((item) => item.classList.remove('active'));
    button.classList.add('active');
    document.querySelector(`#${button.dataset.view}`).classList.add('active');
    document.querySelector('#view-title').textContent = titles[button.dataset.view][0];
    document.querySelector('#view-subtitle').textContent = titles[button.dataset.view][1];
  });
});

document.querySelectorAll('[data-command]').forEach((button) => {
  button.addEventListener('click', () => runMappedCommand(button.dataset.command));
});

document.querySelector('#start-worker').addEventListener('click', async () => renderWorkerStatus(await window.stockPicker.startWorker()));
document.querySelector('#stop-worker').addEventListener('click', async () => renderWorkerStatus(await window.stockPicker.stopWorker()));

window.stockPicker.onCommandLog((payload) => {
  if (!payload.rawOnly) appendLog(payload.text);
});
window.stockPicker.onWorkerStatus((payload) => renderWorkerStatus(payload));
window.stockPicker.onManualTaskStatus((payload) => renderManualTaskStatus(payload));
window.stockPicker.onWorkflowEvent((event) => {
  if (event.ui_log === false) return;
  appendLog(`${formatWorkflowEvent(event)}\n`);
});

window.stockPicker.getSettings().then((settings) => {
  currentSettings = settings;
  setValue('#storage-config', settings.storagePath);
  setValue('#settings-app-base-url', settings.appBaseUrl);
  setValue('#settings-analysis-poll', settings.stockAnalysisPollSeconds);
  setValue('#settings-daily-check', settings.dailyBundleCheckSeconds);
  setValue('#settings-daily-windows', (settings.dailyBundleAllowedWindows || []).join(','));
  setValue('#settings-holding-poll', settings.holdingPricePollSeconds);
  setValue('#settings-log-level', settings.logLevel || 'info');
  checked('#settings-analysis-enabled', settings.stockAnalysisEnabled);
  checked('#settings-daily-enabled', settings.dailyBundleEnabled);
  checked('#settings-holding-enabled', settings.holdingPriceEnabled);
  document.querySelector('#settings-paths').textContent = `Config: ${settings.appWorkerPath} / Env: ${settings.envPath}`;
});
window.stockPicker.getWorkerStatus().then(renderWorkerStatus);
window.stockPicker.getManualTaskStatus().then(renderManualTaskStatus);
window.stockPicker.getRecentLogs().then((text) => {
  if (text) {
    logOutput.textContent = text;
    logOutput.scrollTop = logOutput.scrollHeight;
  }
});
setInterval(() => {
  if (lastWorkerStatus) renderWorkerStatus(lastWorkerStatus);
}, 1000);

async function runMappedCommand(name) {
  if (!currentSettings) currentSettings = await window.stockPicker.getSettings();
  const config = value('#storage-config') || currentSettings.storagePath;
  const workerConfig = currentSettings.appWorkerPath;
  const args = ['--config', config];
  if (name === 'sync-daily-bundle') {
    await run(clean(['app-worker', 'daily-check', '--worker-config', workerConfig, '--auto-pipeline', '--json-events', ...args]), 'sync');
    return;
  }
  if (name === 'stock-analysis') {
    const symbol = value('#manual-analysis-symbol');
    if (!symbol) {
      appendLog(`[${formatDateTime()}] stock_analysis failed: symbol is required\n`);
      return;
    }
    await run(clean(['app-worker', 'analyze-stock', '--worker-config', workerConfig, '--symbol', symbol, '--json-events', ...args]), 'stock_analysis');
    return;
  }
  if (name === 'holding-refresh') {
    await run(clean(['app-worker', 'refresh-holding-prices', '--worker-config', workerConfig, '--json-events', ...args]), 'holding_analysis');
    return;
  }
  if (name === 'save-settings') {
    const settings = await window.stockPicker.saveSettings(readSettings());
    currentSettings = settings;
    appendLog(`[${formatDateTime()}] settings saved: ${settings.appWorkerPath}\n`);
  }
}

async function run(args, manualTaskType) {
  document.querySelector('#top-runtime-state').textContent = 'Running';
  appendLog(`$ stock-picker ${args.join(' ')}\n`);
  const result = await window.stockPicker.runCommand(args, manualTaskType);
  document.querySelector('#top-runtime-state').textContent = result.ok ? 'Idle' : 'Failed';
  appendLog(`[${formatDateTime()}] exit ${result.code}\n`);
}

function clean(items) {
  const output = [];
  for (let index = 0; index < items.length; index += 1) {
    const item = items[index];
    if (item === undefined || item === null || item === '') continue;
    if (String(item).startsWith('--') && (items[index + 1] === '' || items[index + 1] === undefined)) {
      index += 1;
      continue;
    }
    output.push(String(item));
  }
  return output;
}

function value(selector) {
  return document.querySelector(selector)?.value?.trim() || '';
}

function setValue(selector, input) {
  const node = document.querySelector(selector);
  if (node) node.value = input ?? '';
}

function checked(selector, input) {
  const node = document.querySelector(selector);
  if (node) node.checked = Boolean(input);
}

function readSettings() {
  return {
    storagePath: value('#storage-config'),
    appBaseUrl: value('#settings-app-base-url'),
    stockAnalysisPollSeconds: Number(value('#settings-analysis-poll') || 15),
    dailyBundleCheckSeconds: Number(value('#settings-daily-check') || 900),
    dailyBundleAllowedWindows: value('#settings-daily-windows') || '16:00-24:00,00:00-10:00',
    holdingPricePollSeconds: Number(value('#settings-holding-poll') || 300),
    logLevel: value('#settings-log-level') || 'info',
    stockAnalysisEnabled: document.querySelector('#settings-analysis-enabled')?.checked,
    dailyBundleEnabled: document.querySelector('#settings-daily-enabled')?.checked,
    holdingPriceEnabled: document.querySelector('#settings-holding-enabled')?.checked
  };
}

function renderManualTaskStatus(status) {
  if (!status) return;
  lastManualStatus = status;
  document.querySelector('#manual-task-name').textContent = manualTaskName(status.taskType);
  document.querySelector('#manual-task-progress').textContent = `${status.stageIndex || 0}/${status.stageTotal || 0}`;
  document.querySelector('#manual-task-status').textContent = status.status || 'Idle';
  document.querySelector('#manual-task-message').textContent = `${status.stageName || ''}${status.message ? `: ${status.message}` : ''}${status.error ? `\n${status.error}` : ''}`.trim() || 'No manual task is running.';
  updateDuration();
  if (durationTimer) clearInterval(durationTimer);
  if (status.startedAt && !status.endedAt && status.status !== 'idle') {
    durationTimer = setInterval(updateDuration, 1000);
  }
}

function updateDuration() {
  const target = document.querySelector('#manual-task-duration');
  if (!target || !lastManualStatus?.startedAt) {
    target.textContent = '-';
    return;
  }
  const end = lastManualStatus.endedAt ? new Date(lastManualStatus.endedAt).getTime() : Date.now();
  const seconds = Math.max(0, Math.floor((end - new Date(lastManualStatus.startedAt).getTime()) / 1000));
  target.textContent = formatDuration(seconds);
}

function renderWorkerStatus(status) {
  if (!status) return;
  lastWorkerStatus = status;
  const runningText = status.running ? 'Yes' : 'No';
  const statusText = status.running ? status.status : 'Stopped';
  const taskText = status.taskType || 'idle';
  const nextRows = formatNextScheduleRows(status.running ? status.nextSchedules : null);
  const message = `${status.message || ''}${status.lastError ? `\n${status.lastError}` : ''}${status.lastHeartbeatError ? `\nHeartbeat: ${status.lastHeartbeatError}` : ''}`.trim() || (status.running ? 'Worker running.' : 'Worker stopped.');
  const heartbeatStatus = status.lastHeartbeatStatus || 'unknown';
  const heartbeatAt = status.lastHeartbeatAt ? formatDateTime(new Date(status.lastHeartbeatAt)) : '-';

  document.querySelector('#worker-enabled').textContent = runningText;
  document.querySelector('#worker-state').textContent = statusText;
  document.querySelector('#worker-task').textContent = taskText;
  document.querySelector('#worker-heartbeat-status').textContent = heartbeatStatus;
  document.querySelector('#worker-heartbeat-at').textContent = heartbeatAt;
  renderNextRows('#worker-next', nextRows);
  document.querySelector('#worker-message').textContent = message;

  document.querySelector('#worker-page-enabled').textContent = runningText;
  document.querySelector('#worker-page-status').textContent = statusText;
  document.querySelector('#worker-page-task').textContent = taskText;
  document.querySelector('#worker-page-step').textContent = status.step || '-';
  document.querySelector('#worker-page-progress').textContent = `${status.stepIndex || 0}/${status.stepTotal || 0}`;
  document.querySelector('#worker-page-heartbeat-status').textContent = heartbeatStatus;
  document.querySelector('#worker-page-heartbeat-at').textContent = heartbeatAt;
  renderNextRows('#worker-page-next', nextRows);
  document.querySelector('#worker-page-message').textContent = message;
}

function formatNextScheduleRows(nextSchedules) {
  return ['daily_bundle', 'holding_prices', 'stock_analysis'].map((task) => ({
    label: workerTaskName(task),
    value: nextSchedules?.[task] ? `${formatCountdown(nextSchedules[task])} (${formatDateTime(new Date(nextSchedules[task]))})` : '-'
  }));
}

function renderNextRows(selector, rows) {
  const node = document.querySelector(selector);
  if (!node) return;
  node.innerHTML = rows.map((row) => `<div><span>${row.label}</span><strong>${row.value}</strong></div>`).join('');
}

function formatWorkflowEvent(event) {
  const task = manualTaskName(event.task_type || event.workflow_id || 'workflow');
  const progress = event.step_index && event.step_total ? ` ${event.step_index}/${event.step_total}` : '';
  const status = event.status || event.event || 'event';
  const message = event.message || '';
  return `[${formatDateTime()}] ${task}${progress} ${status}: ${message}`;
}

function formatCountdown(value) {
  const diff = new Date(value).getTime() - Date.now();
  if (!Number.isFinite(diff)) return '-';
  if (diff <= 0) return 'now';
  return `in ${formatDuration(Math.ceil(diff / 1000))}`;
}

function formatDuration(seconds) {
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  if (minutes < 60) return `${minutes}m ${rest}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

function formatDateTime(date = new Date()) {
  const input = date instanceof Date ? date : new Date(date);
  if (!Number.isFinite(input.getTime())) return '-';
  const pad = (value) => String(value).padStart(2, '0');
  return `${input.getFullYear()}-${pad(input.getMonth() + 1)}-${pad(input.getDate())} ${pad(input.getHours())}:${pad(input.getMinutes())}:${pad(input.getSeconds())}`;
}

function manualTaskName(taskType) {
  return {
    sync: 'Sync',
    daily_bundle: 'Sync',
    stock_analysis: 'Stock analysis',
    holding_analysis: 'Holding analysis',
    holding_prices: 'Holding analysis',
    manual: 'Manual task'
  }[taskType] || taskType || 'None';
}

function workerTaskName(taskType) {
  return {
    stock_analysis: 'Stock analysis',
    daily_bundle: 'Daily bundle',
    holding_prices: 'Holding prices'
  }[taskType] || taskType;
}

function appendLog(text) {
  logOutput.textContent += text;
  logOutput.scrollTop = logOutput.scrollHeight;
}
