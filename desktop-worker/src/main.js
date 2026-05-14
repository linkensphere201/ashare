const { app, BrowserWindow, Menu, Notification, Tray, ipcMain, nativeImage } = require('electron');
const path = require('node:path');
const fs = require('node:fs');
const { spawn } = require('node:child_process');

let mainWindow;
let tray;
let activeProcess = null;
let workerRunning = false;
let stopAfterCurrent = false;
let workerState = defaultWorkerState();
let timers = [];
const retryState = new Map();
const notificationState = new Map();

const repoRoot = path.resolve(__dirname, '..', '..');
const desktopWorkerRoot = path.resolve(__dirname, '..');

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1220,
    height: 820,
    minWidth: 980,
    minHeight: 680,
    show: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    },
    icon: appIconPath('180.png')
  });

  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  mainWindow.on('close', (event) => {
    if (!app.isQuitting) {
      event.preventDefault();
      mainWindow.hide();
    }
  });
}

function createTray() {
  const trayIconPath = appIconPath('32.png');
  const image = fs.existsSync(trayIconPath)
    ? trayIconPath
    : nativeImage.createFromDataURL('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAGklEQVR4nGNkYGD4z0AEYBxVSF+FAgMDAH5pAhEMZnoXAAAAAElFTkSuQmCC');
  tray = new Tray(image);
  tray.setToolTip('Stock Picker Worker');
  tray.setContextMenu(Menu.buildFromTemplate([
    { label: 'Open Console', click: () => showWindow() },
    { label: 'Start Worker', click: () => startWorker() },
    { label: 'Stop Worker', click: () => stopWorker() },
    { label: 'Stop Current Task', click: () => stopActiveProcess() },
    { type: 'separator' },
    { label: 'Quit', click: () => { app.isQuitting = true; app.quit(); } }
  ]));
  tray.on('double-click', showWindow);
}

function showWindow() {
  if (!mainWindow) createWindow();
  mainWindow.show();
  mainWindow.focus();
}

function portableRoot() {
  if (app.isPackaged) return process.env.PORTABLE_EXECUTABLE_DIR || path.dirname(process.execPath);
  return repoRoot;
}

function resourceRoot() {
  if (app.isPackaged) return process.resourcesPath;
  return repoRoot;
}

function appIconPath(filename) {
  const packagedIcon = path.join(resourceRoot(), 'rabbit-v2-modern', filename);
  if (fs.existsSync(packagedIcon)) return packagedIcon;
  return path.join(desktopWorkerRoot, 'resources', 'rabbit-v2-modern', filename);
}

function configPaths() {
  const root = portableRoot();
  return {
    root,
    appWorker: path.join(root, 'app-worker.yaml'),
    storage: path.join(root, 'storage.yaml'),
    env: path.join(root, '.env'),
    logs: path.join(root, 'logs'),
    state: path.join(root, 'state')
  };
}

function ensurePortableFiles() {
  const paths = configPaths();
  fs.mkdirSync(paths.logs, { recursive: true });
  fs.mkdirSync(paths.state, { recursive: true });
  copyTemplate(paths.appWorker, path.join(resourceRoot(), 'config', 'app-worker.example.yaml'), path.join(repoRoot, 'config', 'app-worker.example.yaml'));
  copyTemplate(paths.storage, path.join(resourceRoot(), 'config', 'storage.example.yaml'), path.join(repoRoot, 'config', 'storage.example.yaml'));
  if (!fs.existsSync(paths.env)) fs.writeFileSync(paths.env, 'STOCK_APP_WORKER_TOKEN=\nTUSHARE_TOKEN=\n', 'utf8');
}

function copyTemplate(target, packagedTemplate, devTemplate) {
  if (fs.existsSync(target)) return;
  const source = fs.existsSync(packagedTemplate) ? packagedTemplate : devTemplate;
  if (fs.existsSync(source)) fs.copyFileSync(source, target);
}

function runtimeCommand() {
  const bundled = path.join(resourceRoot(), 'stock-picker-runtime.exe');
  if (app.isPackaged && fs.existsSync(bundled)) return { command: bundled, args: [] };
  const executable = process.platform === 'win32'
    ? path.join(repoRoot, '.venv', 'Scripts', 'python.exe')
    : path.join(repoRoot, '.venv', 'bin', 'python');
  const fallback = process.platform === 'win32' ? 'python' : 'python3';
  const command = fs.existsSync(executable) ? executable : fallback;
  return { command, args: ['-m', 'stock_picker.cli'] };
}

function loadSettings() {
  ensurePortableFiles();
  const config = parseWorkerYaml(fs.readFileSync(configPaths().appWorker, 'utf8'));
  return {
    appBaseUrl: config.app_base_url || 'http://127.0.0.1:3000',
    stockAnalysisEnabled: config.stock_analysis_enabled !== false,
    dailyBundleEnabled: config.daily_bundle_enabled !== false,
    holdingPriceEnabled: config.holding_price_enabled !== false,
    stockAnalysisPollSeconds: Number(config.poll_interval_seconds || 15),
    dailyBundleCheckSeconds: Number(config.daily_bundle_check_interval_seconds || 900),
    holdingPricePollSeconds: Number(config.holding_price_poll_interval_seconds || 300),
    earliestDailyPublishTime: config.earliest_daily_publish_time || '16:30',
    root: configPaths().root,
    appWorkerPath: configPaths().appWorker,
    storagePath: config.storage_config_path || configPaths().storage,
    envPath: configPaths().env
  };
}

function saveSettings(settings) {
  ensurePortableFiles();
  const yaml = [
    `app_base_url: ${settings.appBaseUrl || 'http://127.0.0.1:3000'}`,
    `storage_config_path: ${settings.storagePath || configPaths().storage}`,
    'worker_id: local-worker',
    'worker_token_env: STOCK_APP_WORKER_TOKEN',
    'tushare_token_env: TUSHARE_TOKEN',
    `poll_interval_seconds: ${Number(settings.stockAnalysisPollSeconds || 15)}`,
    `holding_price_poll_interval_seconds: ${Number(settings.holdingPricePollSeconds || 300)}`,
    `daily_bundle_check_interval_seconds: ${Number(settings.dailyBundleCheckSeconds || 900)}`,
    `earliest_daily_publish_time: "${settings.earliestDailyPublishTime || '16:30'}"`,
    'api_paths:',
    '  analysis_claim: /api/worker/analysis-requests/claim',
    '  analysis_result: /api/worker/analysis-requests/{requestId}/result',
    '  daily_bundle_publish: /api/publish/daily-bundles',
    '  holding_watchlist: /api/worker/holding-prices/watchlist',
    '  holding_prices: /api/worker/holding-prices',
    'tasks:',
    '  stock_analysis:',
    `    enabled: ${settings.stockAnalysisEnabled !== false}`,
    `    poll_interval_seconds: ${Number(settings.stockAnalysisPollSeconds || 15)}`,
    '  daily_bundle:',
    `    enabled: ${settings.dailyBundleEnabled !== false}`,
    `    check_interval_seconds: ${Number(settings.dailyBundleCheckSeconds || 900)}`,
    `    earliest_publish_time: "${settings.earliestDailyPublishTime || '16:30'}"`,
    '    upload_archive_max: 20',
    '  holding_prices:',
    `    enabled: ${settings.holdingPriceEnabled !== false}`,
    `    poll_interval_seconds: ${Number(settings.holdingPricePollSeconds || 300)}`,
    ''
  ].join('\n');
  fs.writeFileSync(configPaths().appWorker, yaml, 'utf8');
  restartWorkerIfRunning();
  return loadSettings();
}

function parseWorkerYaml(text) {
  const config = {};
  let section = '';
  let nested = '';
  for (const rawLine of text.split(/\r?\n/)) {
    if (!rawLine.trim() || rawLine.trim().startsWith('#')) continue;
    const indent = rawLine.match(/^\s*/)[0].length;
    const line = rawLine.trim();
    if (!line.includes(':')) continue;
    const [key, ...rest] = line.split(':');
    const value = rest.join(':').trim().replace(/^["']|["']$/g, '');
    if (indent === 0 && !value) {
      section = key.trim();
      nested = '';
      continue;
    }
    if (indent === 2 && !value) {
      nested = key.trim();
      continue;
    }
    const parsed = parseYamlValue(value);
    if (section === 'api_paths' && indent >= 2) config[apiPathKey(key.trim())] = parsed;
    else if (section === 'tasks' && nested && indent >= 4) config[taskKey(nested, key.trim())] = parsed;
    else if (indent === 0) config[key.trim()] = parsed;
  }
  return config;
}

function parseYamlValue(value) {
  if (value === 'true') return true;
  if (value === 'false') return false;
  if (/^\d+(\.\d+)?$/.test(value)) return Number(value);
  return value;
}

function apiPathKey(key) {
  return {
    analysis_claim: 'analysis_claim_path',
    analysis_result: 'analysis_result_path_template',
    daily_bundle_publish: 'daily_bundle_publish_path',
    holding_watchlist: 'holding_watchlist_path',
    holding_prices: 'holding_prices_path'
  }[key] || key;
}

function taskKey(task, key) {
  const prefix = { stock_analysis: 'stock_analysis', daily_bundle: 'daily_bundle', holding_prices: 'holding_price' }[task] || task;
  if (key === 'enabled') return `${prefix}_enabled`;
  if (task === 'stock_analysis' && key === 'poll_interval_seconds') return 'poll_interval_seconds';
  if (task === 'daily_bundle' && key === 'check_interval_seconds') return 'daily_bundle_check_interval_seconds';
  if (task === 'daily_bundle' && key === 'earliest_publish_time') return 'earliest_daily_publish_time';
  if (task === 'holding_prices' && key === 'poll_interval_seconds') return 'holding_price_poll_interval_seconds';
  return `${prefix}_${key}`;
}

function pythonCommand(args, eventChannel, taskType) {
  if (activeProcess) {
    return Promise.reject(new Error('another workflow is already running'));
  }
  return new Promise((resolve) => {
    const runtime = runtimeCommand();
    const paths = configPaths();
    const env = loadEnv(paths.env);
    setWorkerState({ running: workerRunning, taskType, status: 'running', step: 'starting', message: `stock-picker ${args.join(' ')}`, lastError: null });
    activeProcess = spawn(runtime.command, [...runtime.args, ...args], {
      cwd: app.isPackaged ? paths.root : repoRoot,
      env: { ...process.env, ...env, PYTHONUNBUFFERED: '1' },
      windowsHide: true
    });
    let output = '';
    let errorOutput = '';
    activeProcess.stdout.on('data', (chunk) => {
      const text = chunk.toString();
      output += text;
      appendLog(text);
      mainWindow?.webContents.send(eventChannel, { stream: 'stdout', text });
      parseJsonEvents(text);
    });
    activeProcess.stderr.on('data', (chunk) => {
      const text = chunk.toString();
      errorOutput += text;
      appendLog(text);
      mainWindow?.webContents.send(eventChannel, { stream: 'stderr', text });
    });
    activeProcess.on('close', (code) => {
      activeProcess = null;
      const ok = code === 0;
      setWorkerState({ status: ok ? 'succeeded' : 'failed', taskType, step: 'finished', message: ok ? 'Task completed' : 'Task failed', lastError: ok ? null : errorOutput || output });
      if (ok && Notification.isSupported()) {
        new Notification({ title: 'Stock Picker', body: 'Workflow completed' }).show();
      }
      resolve({ ok, code, output, errorOutput });
    });
  });
}

function parseJsonEvents(text) {
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim().startsWith('{')) continue;
    try {
      const parsed = JSON.parse(line);
      mainWindow?.webContents.send('workflow-event', parsed);
      appendLog(formatWorkflowEvent(parsed));
      notifyWorkflowEvent(parsed);
      setWorkerState({
        taskType: parsed.task_type || parsed.workflow_id || workerState.taskType,
        status: parsed.status || workerState.status,
        step: parsed.step || workerState.step,
        stepIndex: parsed.step_index || workerState.stepIndex,
        stepTotal: parsed.step_total || workerState.stepTotal,
        message: parsed.message || workerState.message,
        lastError: parsed.error || workerState.lastError
      });
    } catch (_) {
      // Keep non-event lines in the plain log.
    }
  }
}

function formatWorkflowEvent(event) {
  const task = event.task_type || event.workflow_id || 'workflow';
  const step = event.step ? ` ${event.step}` : '';
  const progress = event.step_index && event.step_total ? ` ${event.step_index}/${event.step_total}` : '';
  const status = event.status || event.event || 'event';
  const message = event.message || '';
  return `[${new Date().toLocaleTimeString()}] ${task}${step}${progress} ${status}: ${message}\n`;
}

function notifyWorkflowEvent(event) {
  if (!Notification.isSupported()) return;
  const status = String(event.status || event.event || '');
  const key = `${event.task_type || event.workflow_id || 'workflow'}:${event.step || ''}:${status}`;
  const important = /started|running|progress|completed|failed|skipped/.test(status) || /workflow_|step_/.test(String(event.event || ''));
  if (!important) return;
  const now = Date.now();
  const last = notificationState.get(key) || 0;
  const isTerminal = /completed|failed|skipped/.test(status) || /completed|failed/.test(String(event.event || ''));
  if (!isTerminal && now - last < 30000) return;
  notificationState.set(key, now);
  const title = status.includes('failed') || String(event.event || '').includes('failed')
    ? 'Stock Picker task failed'
    : status.includes('completed') || String(event.event || '').includes('completed')
      ? 'Stock Picker task completed'
      : 'Stock Picker task update';
  const step = event.step ? `${event.step}: ` : '';
  const body = `${step}${event.message || status}`.slice(0, 180);
  new Notification({ title, body }).show();
}

function loadEnv(envPath) {
  if (!fs.existsSync(envPath)) return {};
  const values = {};
  for (const line of fs.readFileSync(envPath, 'utf8').split(/\r?\n/)) {
    if (!line.trim() || line.trim().startsWith('#') || !line.includes('=')) continue;
    const [key, ...rest] = line.split('=');
    values[key.trim()] = rest.join('=').trim().replace(/^["']|["']$/g, '');
  }
  return values;
}

function appendLog(text) {
  const paths = configPaths();
  fs.mkdirSync(paths.logs, { recursive: true });
  fs.appendFileSync(path.join(paths.logs, 'worker.log'), text, 'utf8');
}

function startWorker() {
  if (workerRunning) return workerState;
  workerRunning = true;
  stopAfterCurrent = false;
  retryState.clear();
  scheduleAll();
  setWorkerState({ running: true, status: 'idle', message: 'Worker started' });
  return workerState;
}

function stopWorker() {
  stopAfterCurrent = true;
  workerRunning = false;
  clearTimers();
  setWorkerState({ running: false, status: activeProcess ? 'stopping' : 'idle', message: activeProcess ? 'Worker stopping after current task...' : 'Worker stopped' });
  return workerState;
}

function restartWorkerIfRunning() {
  if (!workerRunning) return;
  clearTimers();
  scheduleAll();
}

function scheduleAll() {
  clearTimers();
  const settings = loadSettings();
  if (settings.stockAnalysisEnabled) scheduleTask('stock_analysis', settings.stockAnalysisPollSeconds, () => runAnalysisPoll());
  if (settings.holdingPriceEnabled) scheduleTask('holding_prices', settings.holdingPricePollSeconds, () => runHoldingRefresh());
  if (settings.dailyBundleEnabled) scheduleTask('daily_bundle', settings.dailyBundleCheckSeconds, () => runDailyCheck());
}

function scheduleTask(name, intervalSeconds, fn) {
  const normalDelay = Math.max(1, Number(intervalSeconds || 60));
  const scheduleNext = (delaySeconds) => {
    if (!workerRunning || stopAfterCurrent) return;
    const delay = Math.max(1, Number(delaySeconds || normalDelay));
    const nextRunAt = new Date(Date.now() + delay * 1000).toISOString();
    setNextSchedule(name, nextRunAt);
    const timer = setTimeout(run, delay * 1000);
    timers.push(timer);
  };
  const run = async () => {
    if (!workerRunning || stopAfterCurrent) return;
    if (activeProcess) {
      scheduleNext(Math.min(10, normalDelay));
      return;
    }
    try {
      setNextSchedule(name, null);
      const result = await fn();
      if (!workerRunning || stopAfterCurrent) return;
      if (result && result.ok === false) {
        const text = [result.errorOutput, result.output].filter(Boolean).join('\n');
        const status = classifyError(text);
        const retryDelay = retryDelaySeconds(name, status);
        setWorkerState({ taskType: name, status, lastError: text || 'Task failed', message: status });
        if (retryDelay === null && status !== 'local_data_missing') return;
        scheduleNext(retryDelay ?? normalDelay);
        return;
      }
      retryState.delete(name);
      scheduleNext(normalDelay);
    } catch (error) {
      if (!workerRunning || stopAfterCurrent) return;
      const status = classifyError(error);
      const retryDelay = retryDelaySeconds(name, status);
      setWorkerState({ taskType: name, status, lastError: String(error), message: String(error) });
      if (retryDelay !== null || status === 'local_data_missing') scheduleNext(retryDelay ?? normalDelay);
    }
  };
  setTimeout(run, 500);
}

function clearTimers() {
  for (const timer of timers) clearInterval(timer);
  timers = [];
}

function runAnalysisPoll() {
  return pythonCommand(['app-worker', 'run-once', '--worker-config', configPaths().appWorker, '--config', loadSettings().storagePath], 'command-log', 'stock_analysis');
}

function runHoldingRefresh() {
  return pythonCommand(['app-worker', 'refresh-holding-prices', '--worker-config', configPaths().appWorker, '--config', loadSettings().storagePath], 'command-log', 'holding_prices');
}

function runDailyCheck() {
  const settings = loadSettings();
  if (!isAfterTime(settings.earliestDailyPublishTime)) {
    setWorkerState({ taskType: 'daily_bundle', status: 'waiting_for_publish_window', message: `Waiting for ${settings.earliestDailyPublishTime}` });
    return Promise.resolve({ ok: true, code: 0, output: '', errorOutput: '' });
  }
  return pythonCommand(['app-worker', 'daily-check', '--worker-config', configPaths().appWorker, '--config', settings.storagePath, '--auto-pipeline', '--json-events'], 'command-log', 'daily_bundle');
}

function isAfterTime(value) {
  const [hour, minute] = String(value || '16:30').split(':').map((part) => Number(part));
  const now = new Date();
  const target = new Date();
  target.setHours(hour || 16, minute || 30, 0, 0);
  return now >= target;
}

function classifyError(error) {
  const text = String(error);
  if (/401|403|token/i.test(text)) return 'auth_failed';
  if (/400|validation/i.test(text)) return 'validation_failed';
  if (/5\d\d|network/i.test(text)) return 'network_error';
  if (/Candidate 002 factor run|Tushare|no quote|empty watchlist|local data/i.test(text)) return 'local_data_missing';
  return 'failed';
}

function retryDelaySeconds(taskName, status) {
  if (status === 'auth_failed' || status === 'validation_failed') {
    retryState.delete(taskName);
    return null;
  }
  if (status === 'network_error') {
    const attempts = (retryState.get(taskName) || 0) + 1;
    retryState.set(taskName, attempts);
    if (attempts === 1) return 60;
    if (attempts === 2) return 300;
    return 900;
  }
  retryState.delete(taskName);
  return null;
}

function setNextSchedule(taskName, nextRunAt) {
  const nextSchedules = { ...(workerState.nextSchedules || {}) };
  if (nextRunAt) nextSchedules[taskName] = nextRunAt;
  else delete nextSchedules[taskName];
  setWorkerState({ nextSchedules });
}

function stopActiveProcess() {
  if (!activeProcess) return false;
  activeProcess.kill();
  activeProcess = null;
  setWorkerState({ status: 'interrupted', message: 'Current task interrupted. It can resume from the latest checkpoint.' });
  return true;
}

function defaultWorkerState() {
  return {
    running: false,
    taskType: 'idle',
    status: 'idle',
    step: '',
    stepIndex: 0,
    stepTotal: 0,
    message: '',
    lastError: null,
    nextSchedules: {},
    updatedAt: new Date().toISOString()
  };
}

function setWorkerState(patch) {
  workerState = { ...workerState, ...patch, updatedAt: new Date().toISOString() };
  const paths = configPaths();
  fs.mkdirSync(paths.state, { recursive: true });
  fs.writeFileSync(path.join(paths.state, 'worker-state.json'), JSON.stringify(workerState, null, 2), 'utf8');
  mainWindow?.webContents.send('worker-status', workerState);
}

ipcMain.handle('run-command', async (_event, args) => pythonCommand(args, 'command-log', 'manual'));
ipcMain.handle('stop-command', async () => stopActiveProcess());
ipcMain.handle('start-worker', async () => startWorker());
ipcMain.handle('stop-worker', async () => stopWorker());
ipcMain.handle('get-worker-status', async () => workerState);
ipcMain.handle('get-settings', async () => loadSettings());
ipcMain.handle('save-settings', async (_event, settings) => saveSettings(settings));
ipcMain.handle('show-notification', async (_event, title, body) => {
  if (Notification.isSupported()) new Notification({ title, body }).show();
  return true;
});

app.whenReady().then(() => {
  ensurePortableFiles();
  createWindow();
  createTray();
  app.on('activate', showWindow);
});

app.on('window-all-closed', (event) => {
  event.preventDefault();
});
