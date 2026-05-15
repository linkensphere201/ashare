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
let manualTaskState = defaultManualTaskState();
let activeTaskContext = null;
let timers = [];
const retryState = new Map();
const notificationState = new Map();

const MANUAL_STAGE_DEFINITIONS = {
  sync: [
    ['sync_latest', 'Data sync'],
    ['quality_check', 'Data quality check'],
    ['compute_candidate_002_factors', 'Factor 002 compute'],
    ['rank_candidate_002', 'Candidate pool compute'],
    ['build_daily_bundle', 'Daily Bundle build'],
    ['upload_bundle', 'Bundle upload']
  ],
  stock_analysis: [
    ['validate_symbol', 'Validate symbol'],
    ['resolve_factor_run', 'Resolve latest factor run'],
    ['analyze_stock', 'Generate stock analysis'],
    ['write_summary', 'Write result summary']
  ],
  holding_analysis: [
    ['load_watchlist', 'Load holding watchlist'],
    ['refresh_quotes', 'Refresh holding prices'],
    ['upload_prices', 'Upload and log result']
  ]
};

const repoRoot = path.resolve(__dirname, '..', '..');
const desktopWorkerRoot = path.resolve(__dirname, '..');
const DEFAULT_NO_PROXY = '127.0.0.1,localhost,101.34.212.101';

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
  if (!fs.existsSync(paths.env)) fs.writeFileSync(paths.env, 'APP_API_TOKEN=\nTUSHARE_TOKEN=\n', 'utf8');
  migrateTokenConfig(paths);
}

function copyTemplate(target, packagedTemplate, devTemplate) {
  if (fs.existsSync(target)) return;
  const source = fs.existsSync(packagedTemplate) ? packagedTemplate : devTemplate;
  if (fs.existsSync(source)) fs.copyFileSync(source, target);
}

function migrateTokenConfig(paths) {
  if (fs.existsSync(paths.appWorker)) {
    const original = fs.readFileSync(paths.appWorker, 'utf8');
    const migrated = original.replace(/worker_token_env:\s*STOCK_APP_WORKER_TOKEN/g, 'worker_token_env: APP_API_TOKEN');
    if (migrated !== original) fs.writeFileSync(paths.appWorker, migrated, 'utf8');
  }
  if (fs.existsSync(paths.env)) {
    const text = fs.readFileSync(paths.env, 'utf8');
    const values = parseEnvText(text);
    if (!values.APP_API_TOKEN && values.STOCK_APP_WORKER_TOKEN) {
      fs.writeFileSync(paths.env, `${text.replace(/\s*$/, '')}\nAPP_API_TOKEN=${values.STOCK_APP_WORKER_TOKEN}\n`, 'utf8');
    }
  }
}

function parseEnvText(text) {
  const values = {};
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim() || line.trim().startsWith('#') || !line.includes('=')) continue;
    const [key, ...rest] = line.split('=');
    values[key.trim()] = rest.join('=').trim().replace(/^["']|["']$/g, '');
  }
  return values;
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
    'worker_token_env: APP_API_TOKEN',
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

function pythonCommand(args, eventChannel, taskType, manualTaskType = null) {
  if (activeProcess) {
    return Promise.reject(new Error('another workflow is already running'));
  }
  return new Promise((resolve) => {
    const runtime = runtimeCommand();
    const paths = configPaths();
    const env = loadEnv(paths.env);
    activeTaskContext = { taskType, manualTaskType, lastStageKey: null, lastProgressLogAt: 0, lastProgressMessage: '' };
    if (!manualTaskType) setWorkerState({ running: workerRunning, taskType, status: 'running', step: 'starting', message: `stock-picker ${args.join(' ')}`, lastError: null });
    if (manualTaskType) {
      setManualTaskState({
        taskType: manualTaskType,
        status: 'running',
        stageKey: 'starting',
        stageName: 'Starting',
        stageIndex: 0,
        stageTotal: manualStageTotal(manualTaskType),
        message: `stock-picker ${args.join(' ')}`,
        error: null,
        startedAt: new Date().toISOString(),
        endedAt: null
      });
    }
    activeProcess = spawn(runtime.command, [...runtime.args, ...args], {
      cwd: app.isPackaged ? paths.root : repoRoot,
      env: runtimeEnv(env),
      windowsHide: true
    });
    let output = '';
    let errorOutput = '';
    activeProcess.stdout.on('data', (chunk) => {
      const text = chunk.toString();
      output += text;
      appendLog(text);
      if (!manualTaskType) mainWindow?.webContents.send(eventChannel, { stream: 'stdout', text });
      else if (!isJsonOnlyChunk(text)) mainWindow?.webContents.send(eventChannel, { stream: 'stdout', text, rawOnly: true });
      parseJsonEvents(text, activeTaskContext);
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
      if (!manualTaskType) setWorkerState({ status: ok ? 'succeeded' : 'failed', taskType, step: 'finished', message: ok ? 'Task completed' : 'Task failed', lastError: ok ? null : errorOutput || output });
      if (manualTaskType) {
        const terminal = ok ? terminalManualStage(manualTaskType) : manualTaskState;
        setManualTaskState({
          ...terminal,
          taskType: manualTaskType,
          status: ok ? 'completed' : 'failed',
          message: ok ? 'Task completed' : 'Task failed',
          error: ok ? null : (errorOutput || output).slice(-2000),
          endedAt: new Date().toISOString()
        });
      }
      activeTaskContext = null;
      if (ok && Notification.isSupported()) {
        new Notification({ title: 'Stock Picker', body: 'Workflow completed' }).show();
      }
      resolve({ ok, code, output, errorOutput });
    });
  });
}

function runtimeEnv(localEnv) {
  const noProxy = mergeNoProxy(localEnv.NO_PROXY || process.env.NO_PROXY, localEnv.no_proxy || process.env.no_proxy);
  const merged = {
    ...process.env,
    ...localEnv,
    PYTHONUNBUFFERED: '1',
    NO_PROXY: noProxy,
    no_proxy: noProxy
  };
  for (const key of ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']) {
    delete merged[key];
  }
  return merged;
}

function mergeNoProxy(...values) {
  const entries = [];
  for (const value of [...values, DEFAULT_NO_PROXY]) {
    if (!value) continue;
    for (const item of String(value).split(',')) {
      const trimmed = item.trim();
      if (trimmed && !entries.includes(trimmed)) entries.push(trimmed);
    }
  }
  return entries.join(',');
}

function parseJsonEvents(text, context = null) {
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim().startsWith('{')) continue;
    try {
      const parsed = JSON.parse(line);
      const uiEvent = normalizeUiEvent(parsed, context);
      mainWindow?.webContents.send('workflow-event', uiEvent);
      if (uiEvent.ui_log !== false) {
        appendLog(formatWorkflowEvent(uiEvent));
        if (uiEvent.ui_notify !== false) notifyWorkflowEvent(uiEvent);
      }
      if (context?.manualTaskType) updateManualTaskFromEvent(uiEvent, context);
      if (!context?.manualTaskType) {
        setWorkerState({
          taskType: parsed.task_type || parsed.workflow_id || workerState.taskType,
          status: parsed.status || workerState.status,
          step: parsed.step || workerState.step,
          stepIndex: parsed.step_index || workerState.stepIndex,
          stepTotal: parsed.step_total || workerState.stepTotal,
          message: parsed.message || workerState.message,
          lastError: parsed.error || workerState.lastError
        });
      }
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

function normalizeUiEvent(event, context) {
  if (!context?.manualTaskType) return event;
  const mapped = mapManualStage(context.manualTaskType, event);
  if (!mapped) return { ...event, ui_log: false };
  const status = mapped.status || event.status || event.event || 'running';
  const stageChanged = context.lastStageKey !== mapped.stageKey;
  const terminal = /completed|failed|skipped/.test(String(status)) || /completed|failed/.test(String(event.event || ''));
  const progressLog = shouldLogManualProgress(event, context);
  context.lastStageKey = mapped.stageKey;
  return {
    ...event,
    task_type: context.manualTaskType,
    step: mapped.stageKey,
    step_index: mapped.stageIndex,
    step_total: mapped.stageTotal,
    status,
    message: mapped.message || event.message || mapped.stageName,
    ui_log: stageChanged || terminal || progressLog,
    ui_notify: stageChanged || terminal
  };
}

function shouldLogManualProgress(event, context) {
  if (context.manualTaskType !== 'sync') return false;
  const eventName = String(event.event || '');
  if (eventName !== 'step_progress') return false;
  const message = String(event.message || '');
  const now = Date.now();
  if (message && message !== context.lastProgressMessage) {
    context.lastProgressMessage = message;
    context.lastProgressLogAt = now;
    return true;
  }
  if (now - context.lastProgressLogAt >= 10000) {
    context.lastProgressLogAt = now;
    return true;
  }
  return false;
}

function mapManualStage(taskType, event) {
  const step = String(event.step || '');
  const status = String(event.status || event.event || '');
  if (taskType === 'sync') {
    if (step === 'upload_bundle') return manualStage(taskType, 'upload_bundle', event);
    if (step === 'build_bundle') return manualStage(taskType, 'build_daily_bundle', event);
    if (step === 'sync_latest' || step === 'sync_dry_run' || /sync/.test(step)) return manualStage(taskType, 'sync_latest', event);
    if (step === 'quality_check') return manualStage(taskType, 'quality_check', event);
    if (step === 'compute_candidate_002_factors') return manualStage(taskType, 'compute_candidate_002_factors', event);
    if (step === 'rank_candidate_002' || step === 'backtest_candidate_002' || step === 'create_snapshot') return manualStage(taskType, 'rank_candidate_002', event);
    if (step === 'build_daily_bundle' || /build/.test(step)) return manualStage(taskType, 'build_daily_bundle', event);
    if (/failed|completed/.test(status)) return terminalManualStage(taskType, event);
    return null;
  }
  if (taskType === 'stock_analysis') {
    if (step === 'validate_symbol') return manualStage(taskType, 'validate_symbol', event);
    if (step === 'resolve_factor_run') return manualStage(taskType, 'resolve_factor_run', event);
    if (step === 'analyze_stock') return manualStage(taskType, 'analyze_stock', event);
    if (step === 'write_summary' || /completed|failed/.test(status)) return manualStage(taskType, 'write_summary', event);
    return null;
  }
  if (taskType === 'holding_analysis') {
    if (step === 'load_watchlist') return manualStage(taskType, 'load_watchlist', event);
    if (step === 'refresh_quotes') return manualStage(taskType, 'refresh_quotes', event);
    if (step === 'upload_prices' || /completed|failed/.test(status)) return manualStage(taskType, 'upload_prices', event);
    return null;
  }
  return null;
}

function manualStage(taskType, stageKey, event = {}) {
  const stages = MANUAL_STAGE_DEFINITIONS[taskType] || [];
  const index = stages.findIndex(([key]) => key === stageKey);
  return {
    stageKey,
    stageName: index >= 0 ? stages[index][1] : stageKey,
    stageIndex: index >= 0 ? index + 1 : 0,
    stageTotal: stages.length,
    status: event.status || event.event || 'running',
    message: event.message
  };
}

function terminalManualStage(taskType, event = {}) {
  const stages = MANUAL_STAGE_DEFINITIONS[taskType] || [];
  const [stageKey, stageName] = stages[stages.length - 1] || ['finished', 'Finished'];
  return {
    stageKey,
    stageName,
    stageIndex: stages.length,
    stageTotal: stages.length,
    status: event.status || event.event || 'completed',
    message: event.message
  };
}

function updateManualTaskFromEvent(event, context) {
  if (event.ui_log === false) return;
  setManualTaskState({
    taskType: context.manualTaskType,
    status: event.status || event.event || manualTaskState.status,
    stageKey: event.step || manualTaskState.stageKey,
    stageName: manualStage(context.manualTaskType, event.step || '').stageName || event.step || manualTaskState.stageName,
    stageIndex: event.step_index || manualTaskState.stageIndex,
    stageTotal: event.step_total || manualTaskState.stageTotal,
    message: event.message || manualTaskState.message,
    error: event.error || manualTaskState.error
  });
}

function manualStageTotal(taskType) {
  return (MANUAL_STAGE_DEFINITIONS[taskType] || []).length;
}

function isJsonOnlyChunk(text) {
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  return lines.length > 0 && lines.every((line) => line.trim().startsWith('{'));
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

function appendUiLog(message) {
  const line = `[${new Date().toLocaleTimeString()}] ${message}\n`;
  appendLog(line);
  mainWindow?.webContents.send('command-log', { stream: 'worker', text: line });
}

function readRecentLogs(maxBytes = 40000) {
  const logPath = path.join(configPaths().logs, 'worker.log');
  if (!fs.existsSync(logPath)) return '';
  const stats = fs.statSync(logPath);
  const start = Math.max(0, stats.size - maxBytes);
  const handle = fs.openSync(logPath, 'r');
  try {
    const buffer = Buffer.alloc(stats.size - start);
    fs.readSync(handle, buffer, 0, buffer.length, start);
    return buffer.toString('utf8');
  } finally {
    fs.closeSync(handle);
  }
}

function startWorker() {
  if (workerRunning) return workerState;
  workerRunning = true;
  stopAfterCurrent = false;
  retryState.clear();
  appendUiLog('worker started');
  scheduleAll();
  setWorkerState({ running: true, status: 'idle', message: 'Worker started' });
  return workerState;
}

function stopWorker() {
  stopAfterCurrent = true;
  workerRunning = false;
  clearTimers();
  appendUiLog(activeProcess ? 'worker stop requested; current task will finish first' : 'worker stopped');
  setWorkerState({ running: false, status: activeProcess ? 'stopping' : 'idle', message: activeProcess ? 'Worker stopping after current task...' : 'Worker stopped', nextSchedules: {} });
  return workerState;
}

function restartWorkerIfRunning() {
  if (!workerRunning) return;
  appendUiLog('worker settings changed; rescheduling tasks');
  clearTimers();
  scheduleAll();
}

function scheduleAll() {
  clearTimers();
  const settings = loadSettings();
  appendUiLog(`worker schedule loaded: stock_analysis=${settings.stockAnalysisEnabled ? `${settings.stockAnalysisPollSeconds}s` : 'disabled'}, daily_bundle=${settings.dailyBundleEnabled ? `${settings.dailyBundleCheckSeconds}s` : 'disabled'}, holding_prices=${settings.holdingPriceEnabled ? `${settings.holdingPricePollSeconds}s` : 'disabled'}`);
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
    appendUiLog(`${workerTaskLabel(name)} scheduled in ${delay}s at ${new Date(nextRunAt).toLocaleTimeString()}`);
    const timer = setTimeout(run, delay * 1000);
    timers.push(timer);
  };
  const run = async () => {
    if (!workerRunning || stopAfterCurrent) return;
    if (activeProcess) {
      appendUiLog(`${workerTaskLabel(name)} delayed because another task is running`);
      scheduleNext(Math.min(10, normalDelay));
      return;
    }
    try {
      setNextSchedule(name, null);
      appendUiLog(`${workerTaskLabel(name)} started`);
      const result = await fn();
      if (!workerRunning || stopAfterCurrent) return;
      if (result && result.ok === false) {
        const text = [result.errorOutput, result.output].filter(Boolean).join('\n');
        const status = classifyError(text);
        const retryDelay = retryDelaySeconds(name, status);
        appendUiLog(`${workerTaskLabel(name)} failed: ${status}${retryDelay !== null ? `; retry in ${retryDelay}s` : ''}`);
        setWorkerState({ taskType: name, status, lastError: text || 'Task failed', message: status });
        if (retryDelay === null && status !== 'local_data_missing') return;
        scheduleNext(retryDelay ?? normalDelay);
        return;
      }
      retryState.delete(name);
      appendUiLog(`${workerTaskLabel(name)} completed`);
      scheduleNext(normalDelay);
    } catch (error) {
      if (!workerRunning || stopAfterCurrent) return;
      const status = classifyError(error);
      const retryDelay = retryDelaySeconds(name, status);
      appendUiLog(`${workerTaskLabel(name)} failed: ${String(error)}${retryDelay !== null ? `; retry in ${retryDelay}s` : ''}`);
      setWorkerState({ taskType: name, status, lastError: String(error), message: String(error) });
      if (retryDelay !== null || status === 'local_data_missing') scheduleNext(retryDelay ?? normalDelay);
    }
  };
  appendUiLog(`${workerTaskLabel(name)} initial check scheduled now`);
  setTimeout(run, 500);
}

function clearTimers() {
  for (const timer of timers) clearInterval(timer);
  timers = [];
}

function runAnalysisPoll() {
  const settings = loadSettings();
  appendUiLog(`Stock analysis queue target: ${settings.appBaseUrl}/api/worker/analysis-requests/claim`);
  return pythonCommand(['app-worker', 'run-once', '--worker-config', configPaths().appWorker, '--config', settings.storagePath], 'command-log', 'stock_analysis');
}

function runHoldingRefresh() {
  const settings = loadSettings();
  appendUiLog(`Holding prices target: ${settings.appBaseUrl}/api/worker/holding-prices/watchlist`);
  return pythonCommand(['app-worker', 'refresh-holding-prices', '--worker-config', configPaths().appWorker, '--config', settings.storagePath], 'command-log', 'holding_prices');
}

function runDailyCheck() {
  const settings = loadSettings();
  if (!isAfterTime(settings.earliestDailyPublishTime)) {
    appendUiLog(`Daily bundle skipped: waiting for publish window ${settings.earliestDailyPublishTime}`);
    setWorkerState({ taskType: 'daily_bundle', status: 'waiting_for_publish_window', message: `Waiting for ${settings.earliestDailyPublishTime}` });
    return Promise.resolve({ ok: true, code: 0, output: '', errorOutput: '' });
  }
  return pythonCommand(['app-worker', 'daily-check', '--worker-config', configPaths().appWorker, '--config', settings.storagePath, '--auto-pipeline', '--json-events'], 'command-log', 'daily_bundle');
}

function workerTaskLabel(taskName) {
  return {
    stock_analysis: 'Stock analysis queue',
    daily_bundle: 'Daily bundle',
    holding_prices: 'Holding prices'
  }[taskName] || taskName;
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
  if (activeTaskContext?.manualTaskType) {
    setManualTaskState({
      status: 'interrupted',
      message: 'Current task interrupted. It can resume from the latest checkpoint.',
      endedAt: new Date().toISOString()
    });
  } else {
    setWorkerState({ status: 'interrupted', message: 'Current task interrupted. It can resume from the latest checkpoint.' });
  }
  activeTaskContext = null;
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

function defaultManualTaskState() {
  return {
    taskType: 'none',
    status: 'idle',
    stageKey: '',
    stageName: '',
    stageIndex: 0,
    stageTotal: 0,
    message: 'No manual task is running.',
    error: null,
    startedAt: null,
    endedAt: null,
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

function setManualTaskState(patch) {
  manualTaskState = { ...manualTaskState, ...patch, updatedAt: new Date().toISOString() };
  const paths = configPaths();
  fs.mkdirSync(paths.state, { recursive: true });
  fs.writeFileSync(path.join(paths.state, 'manual-task-state.json'), JSON.stringify(manualTaskState, null, 2), 'utf8');
  mainWindow?.webContents.send('manual-task-status', manualTaskState);
}

ipcMain.handle('run-command', async (_event, args, manualTaskType) => pythonCommand(args, 'command-log', 'manual', manualTaskType || 'manual'));
ipcMain.handle('stop-command', async () => stopActiveProcess());
ipcMain.handle('start-worker', async () => startWorker());
ipcMain.handle('stop-worker', async () => stopWorker());
ipcMain.handle('get-worker-status', async () => workerState);
ipcMain.handle('get-manual-task-status', async () => manualTaskState);
ipcMain.handle('get-recent-logs', async () => readRecentLogs());
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
