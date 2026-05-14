const titles = {
  home: ['Home', 'Local workflow status and latest outputs.'],
  sync: ['Sync', 'Preflight local data gaps, then run confirmed workflows.'],
  report: ['Daily Bundle', 'Build and upload market status plus candidate pool bundles.'],
  analysis: ['Stock Analysis', 'Generate a structured research card for one symbol.'],
  worker: ['Worker', 'Poll stock-analysis jobs, publish daily bundles, and refresh holding prices.'],
  logs: ['Logs', 'Workflow and command output.'],
  settings: ['Settings', 'Local runtime settings.']
};

const logOutput = document.querySelector('#log-output');
const runtimeState = document.querySelector('#runtime-state');
const latestWorkflow = document.querySelector('#latest-workflow');
const workerState = document.querySelector('#worker-state');

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

document.querySelector('#stop-command').addEventListener('click', async () => {
  const stopped = await window.stockPicker.stopCommand();
  appendLog(stopped ? 'Stopped current task.\n' : 'No active task.\n');
});
document.querySelector('#start-worker').addEventListener('click', async () => renderWorkerStatus(await window.stockPicker.startWorker()));
document.querySelector('#stop-worker').addEventListener('click', async () => renderWorkerStatus(await window.stockPicker.stopWorker()));

window.stockPicker.onCommandLog((payload) => appendLog(payload.text));
window.stockPicker.onWorkerStatus((payload) => renderWorkerStatus(payload));
window.stockPicker.onWorkflowEvent((event) => {
  latestWorkflow.textContent = `${event.workflow_id || 'workflow'} / ${event.status || event.event}`;
  appendLog(`[event] ${event.event}: ${event.message}\n`);
});

window.stockPicker.getSettings().then((settings) => {
  setValue('#settings-app-base-url', settings.appBaseUrl);
  setValue('#settings-worker-token-env', settings.workerTokenEnv);
  setValue('#settings-publisher-token-env', settings.publisherTokenEnv);
  setValue('#settings-tushare-token-env', settings.tushareTokenEnv);
  setValue('#settings-analysis-claim', settings.analysisClaimPath);
  setValue('#settings-analysis-result', settings.analysisResultPath);
  setValue('#settings-daily-publish', settings.dailyBundlePublishPath);
  setValue('#settings-holding-watchlist', settings.holdingWatchlistPath);
  setValue('#settings-holding-prices', settings.holdingPricesPath);
  setValue('#settings-analysis-poll', settings.stockAnalysisPollSeconds);
  setValue('#settings-daily-check', settings.dailyBundleCheckSeconds);
  setValue('#settings-daily-time', settings.earliestDailyPublishTime);
  setValue('#settings-holding-poll', settings.holdingPricePollSeconds);
  checked('#settings-analysis-enabled', settings.stockAnalysisEnabled);
  checked('#settings-daily-enabled', settings.dailyBundleEnabled);
  checked('#settings-holding-enabled', settings.holdingPriceEnabled);
  document.querySelector('#settings-paths').textContent = `Config: ${settings.appWorkerPath} / Env: ${settings.envPath}`;
});
window.stockPicker.getWorkerStatus().then(renderWorkerStatus);

async function runMappedCommand(name) {
  const config = value('#storage-config') || 'config/storage.yaml';
  const args = ['--config', config];
  if (name === 'workflow-status') {
    await run(['workflow', 'status', ...args], '#status-output');
    return;
  }
  if (name === 'sync-dry-run') {
    await run(clean(['workflow', 'sync-report', '--dry-run', '--json-events', '--start-date', value('#sync-start'), '--end-date', value('#sync-end'), '--factor-run-id', value('#sync-factor-run'), '--top', value('#sync-top'), ...args]));
    return;
  }
  if (name === 'sync-confirm') {
    await run(clean(['workflow', 'sync-report', '--confirm', '--json-events', '--start-date', value('#sync-start'), '--end-date', value('#sync-end'), '--factor-run-id', value('#sync-factor-run'), '--top', value('#sync-top'), ...args]));
    return;
  }
  if (name === 'build-artifact') {
    await run(clean(['publish', 'build-daily-bundle', '--factor-run-id', value('#report-factor-run'), '--trade-date', value('#report-date'), '--output', value('#report-output'), '--top', value('#report-top'), ...args]));
    return;
  }
  if (name === 'mock-upload') {
    await run(clean(['app-worker', 'daily-check', '--worker-config', value('#worker-config'), '--factor-run-id', value('#report-factor-run'), '--trade-date', value('#report-date'), '--top', value('#report-top'), '--mock-upload', ...args]));
    await window.stockPicker.notify('Stock Picker', 'Daily bundle mock upload completed');
    return;
  }
  if (name === 'stock-analysis') {
    await run(clean(['analysis', 'stock', '--symbol', value('#analysis-symbol'), '--factor-run-id', value('#analysis-factor-run'), '--trade-date', value('#analysis-date'), ...args]));
    return;
  }
  if (name === 'worker-once') {
    await run(clean(['app-worker', 'run-once', '--worker-config', value('#worker-config'), '--mock-task', value('#mock-task'), ...args]));
    return;
  }
  if (name === 'worker-loop') {
    await run(clean(['app-worker', 'run', '--max-iterations', '1', '--worker-config', value('#worker-config'), ...args]));
    return;
  }
  if (name === 'daily-check') {
    await run(clean(['app-worker', 'daily-check', '--worker-config', value('#worker-config'), '--factor-run-id', value('#daily-factor-run'), '--trade-date', value('#daily-trade-date'), '--top', value('#daily-top'), '--mock-upload', ...args]));
    return;
  }
  if (name === 'holding-refresh') {
    await run(clean(['app-worker', 'refresh-holding-prices', '--worker-config', value('#worker-config'), '--trade-date', value('#holding-trade-date'), '--token-env', value('#holding-token-env'), '--mock-watchlist', value('#holding-watchlist'), '--mock-upload-path', value('#holding-upload-path'), ...args]));
    return;
  }
  if (name === 'save-settings') {
    const settings = await window.stockPicker.saveSettings(readSettings());
    appendLog(`Saved settings: ${settings.appWorkerPath}\n`);
  }
}

async function run(args, outputSelector) {
  runtimeState.textContent = 'Running';
  appendLog(`$ stock-picker ${args.join(' ')}\n`);
  const result = await window.stockPicker.runCommand(args);
  runtimeState.textContent = result.ok ? 'Idle' : 'Failed';
  if (outputSelector) document.querySelector(outputSelector).textContent = result.output || result.errorOutput;
  appendLog(`\n[exit ${result.code}]\n`);
}

function clean(items) {
  const output = [];
  for (let index = 0; index < items.length; index += 1) {
    const item = items[index];
    if (item === undefined || item === null || item === '') {
      index += item === '' && String(items[index - 1] || '').startsWith('--') ? 0 : 0;
      continue;
    }
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
    appBaseUrl: value('#settings-app-base-url'),
    workerTokenEnv: value('#settings-worker-token-env'),
    publisherTokenEnv: value('#settings-publisher-token-env'),
    tushareTokenEnv: value('#settings-tushare-token-env'),
    analysisClaimPath: value('#settings-analysis-claim'),
    analysisResultPath: value('#settings-analysis-result'),
    dailyBundlePublishPath: value('#settings-daily-publish'),
    holdingWatchlistPath: value('#settings-holding-watchlist'),
    holdingPricesPath: value('#settings-holding-prices'),
    stockAnalysisPollSeconds: Number(value('#settings-analysis-poll') || 15),
    dailyBundleCheckSeconds: Number(value('#settings-daily-check') || 900),
    earliestDailyPublishTime: value('#settings-daily-time') || '16:30',
    holdingPricePollSeconds: Number(value('#settings-holding-poll') || 300),
    stockAnalysisEnabled: document.querySelector('#settings-analysis-enabled')?.checked,
    dailyBundleEnabled: document.querySelector('#settings-daily-enabled')?.checked,
    holdingPriceEnabled: document.querySelector('#settings-holding-enabled')?.checked
  };
}

function renderWorkerStatus(status) {
  if (!status) return;
  workerState.textContent = status.running ? status.status : 'Stopped';
  document.querySelector('#worker-task').textContent = status.taskType || 'idle';
  document.querySelector('#worker-step').textContent = status.step || '-';
  document.querySelector('#worker-progress').textContent = `${status.stepIndex || 0}/${status.stepTotal || 0}`;
  document.querySelector('#worker-next').textContent = formatNextSchedules(status.nextSchedules);
  document.querySelector('#worker-message').textContent = `${status.message || ''}\n${status.lastError || ''}`.trim();
}

function formatNextSchedules(nextSchedules) {
  const entries = Object.entries(nextSchedules || {});
  if (!entries.length) return '-';
  return entries
    .map(([task, time]) => `${task}: ${new Date(time).toLocaleTimeString()}`)
    .join(' / ');
}

function appendLog(text) {
  logOutput.textContent += text;
  logOutput.scrollTop = logOutput.scrollHeight;
}
