const titles = {
  home: ['Home', 'Local workflow status and latest outputs.'],
  sync: ['Sync', 'Preflight local data gaps, then run confirmed workflows.'],
  report: ['Daily Bundle', 'Build and upload market status plus candidate pool bundles.'],
  analysis: ['Stock Analysis', 'Generate a structured research card for one symbol.'],
  worker: ['Worker', 'Poll stock-analysis jobs and run daily bundle checks.'],
  logs: ['Logs', 'Workflow and command output.'],
  settings: ['Settings', 'Local runtime settings.']
};

const logOutput = document.querySelector('#log-output');
const runtimeState = document.querySelector('#runtime-state');
const latestWorkflow = document.querySelector('#latest-workflow');

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

window.stockPicker.onCommandLog((payload) => appendLog(payload.text));
window.stockPicker.onWorkflowEvent((event) => {
  latestWorkflow.textContent = `${event.workflow_id || 'workflow'} / ${event.status || event.event}`;
  appendLog(`[event] ${event.event}: ${event.message}\n`);
});

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

function appendLog(text) {
  logOutput.textContent += text;
  logOutput.scrollTop = logOutput.scrollHeight;
}
