const { app, BrowserWindow, Menu, Notification, Tray, ipcMain, nativeImage } = require('electron');
const path = require('node:path');
const fs = require('node:fs');
const http = require('node:http');
const https = require('node:https');
const { spawn } = require('node:child_process');
const {
  LOG_LEVELS,
  mergeSettingsIntoWorkerYaml,
  normalizeAllowedWindows,
  normalizeLogLevel,
  parseWorkerYaml
} = require('./worker-config');
const { buildApiUrl, sanitizeHeartbeatMessage, workerCapabilities } = require('./heartbeat-utils');
const { isWithinAnyWindow, localDateKey, nextWindowStart } = require('./scheduler-utils');
const { runtimeEnv } = require('./python-runner');
const {
  eventLogLevel,
  formatWorkflowEvent,
  isJsonOnlyChunk,
  manualStage,
  manualStageTotal,
  normalizeUiEvent
} = require('./event-utils');

let mainWindow;
let tray;
let activeProcess = null;
let workerRunning = false;
let stopAfterCurrent = false;
let workerState = defaultWorkerState();
let manualTaskState = defaultManualTaskState();
let activeTaskContext = null;
let timers = [];
let heartbeatTimer = null;
const retryState = new Map();
const notificationState = new Map();
const automaticJobState = loadAutomaticJobState();

const AUTOMATIC_JOBS = {
  daily_bundle: {
    label: 'Daily bundle',
    mode: 'daily',
    enabledKey: 'dailyBundleEnabled',
    intervalKey: 'dailyBundleCheckSeconds',
    windowsKey: 'dailyBundleAllowedWindows',
    run: () => runDailyCheck()
  },
  holding_prices: {
    label: 'Holding prices',
    mode: 'periodic',
    enabledKey: 'holdingPriceEnabled',
    intervalKey: 'holdingPricePollSeconds',
    run: () => runHoldingRefresh()
  },
  stock_analysis: {
    label: 'Stock analysis',
    mode: 'periodic',
    enabledKey: 'stockAnalysisEnabled',
    intervalKey: 'stockAnalysisPollSeconds',
    run: () => runAnalysisPoll()
  }
};

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
    workerStatusIntervalSeconds: Number(config.worker_status_interval_seconds || 30),
    dailyBundleAllowedWindows: normalizeAllowedWindows(config.daily_bundle_allowed_windows),
    earliestDailyPublishTime: config.earliest_daily_publish_time || '16:30',
    logLevel: normalizeLogLevel(config.log_level || 'info'),
    workerId: config.worker_id || 'local-worker',
    workerTokenEnv: config.worker_token_env || 'APP_API_TOKEN',
    workerStatusPath: config.worker_status_path || '/api/worker/status',
    root: configPaths().root,
    appWorkerPath: configPaths().appWorker,
    storagePath: config.storage_config_path || configPaths().storage,
    envPath: configPaths().env
  };
}

function saveSettings(settings) {
  ensurePortableFiles();
  const paths = configPaths();
  const original = fs.readFileSync(paths.appWorker, 'utf8');
  const yaml = mergeSettingsIntoWorkerYaml(original, settings, paths.storage);
  fs.writeFileSync(configPaths().appWorker, yaml, 'utf8');
  restartWorkerIfRunning();
  return loadSettings();
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
        stageTotal: manualStageTotal(manualTaskType, MANUAL_STAGE_DEFINITIONS),
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
      const jsonOnly = isJsonOnlyChunk(text);
      if (!jsonOnly) writeLog(manualTaskType ? 'info' : 'debug', text, { emitUi: !manualTaskType, stream: 'stdout' });
      if (manualTaskType && !jsonOnly && shouldLog('info')) mainWindow?.webContents.send(eventChannel, { stream: 'stdout', text, rawOnly: true });
      parseJsonEvents(text, activeTaskContext);
    });
    activeProcess.stderr.on('data', (chunk) => {
      const text = chunk.toString();
      errorOutput += text;
      writeLog('error', text, { emitUi: true, stream: 'stderr' });
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

function parseJsonEvents(text, context = null) {
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim().startsWith('{')) continue;
    try {
      const parsed = JSON.parse(line);
      const uiEvent = normalizeUiEvent(parsed, context, AUTOMATIC_JOBS, MANUAL_STAGE_DEFINITIONS);
      const level = eventLogLevel(uiEvent);
      const shouldEmitLog = uiEvent.ui_log !== false && shouldLog(level);
      mainWindow?.webContents.send('workflow-event', { ...uiEvent, ui_log: shouldEmitLog });
      if (uiEvent.ui_log !== false) {
        writeLog(level, formatWorkflowEvent(uiEvent, formatDateTime()), { emitUi: true });
        if (shouldEmitLog && uiEvent.ui_notify !== false) notifyWorkflowEvent(uiEvent);
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

function updateManualTaskFromEvent(event, context) {
  if (event.ui_log === false) return;
  setManualTaskState({
    taskType: context.manualTaskType,
    status: event.status || event.event || manualTaskState.status,
    stageKey: event.step || manualTaskState.stageKey,
    stageName: manualStage(context.manualTaskType, event.step || '', {}, MANUAL_STAGE_DEFINITIONS).stageName || event.step || manualTaskState.stageName,
    stageIndex: event.step_index || manualTaskState.stageIndex,
    stageTotal: event.step_total || manualTaskState.stageTotal,
    message: event.message || manualTaskState.message,
    error: event.error || manualTaskState.error
  });
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

function notifyAutomaticTask(taskName, level, message) {
  if (!Notification.isSupported()) return;
  const now = Date.now();
  const normalizedLevel = level === 'error' ? 'error' : 'info';
  const throttleMs = normalizedLevel === 'error' ? 60 * 1000 : 10 * 60 * 1000;
  const key = `automatic:${taskName}:${normalizedLevel}`;
  const last = notificationState.get(key) || 0;
  if (now - last < throttleMs) return;
  notificationState.set(key, now);
  new Notification({
    title: normalizedLevel === 'error' ? 'Stock Picker automatic task failed' : 'Stock Picker automatic task update',
    body: `${workerTaskLabel(taskName)}: ${message}`.slice(0, 180)
  }).show();
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

function shouldLog(level) {
  const configured = normalizeLogLevel(loadSettings().logLevel);
  return LOG_LEVELS[normalizeLogLevel(level)] >= LOG_LEVELS[configured];
}

function appendLog(text) {
  const paths = configPaths();
  fs.mkdirSync(paths.logs, { recursive: true });
  fs.appendFileSync(path.join(paths.logs, 'worker.log'), text, 'utf8');
}

function writeLog(level, text, options = {}) {
  if (!shouldLog(level)) return;
  appendLog(text);
  if (options.emitUi) mainWindow?.webContents.send('command-log', { stream: options.stream || 'worker', text });
}

function appendUiLog(message, level = 'info') {
  const line = `[${formatDateTime()}] ${message}\n`;
  writeLog(level, line, { emitUi: true });
}

function formatDateTime(date = new Date()) {
  const pad = (value) => String(value).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
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
  startHeartbeat();
  scheduleAll();
  setWorkerState({ running: true, status: 'idle', message: 'Worker started' });
  return workerState;
}

function stopWorker() {
  stopAfterCurrent = true;
  workerRunning = false;
  clearTimers();
  stopHeartbeat();
  appendUiLog(activeProcess ? 'worker stop requested; current task will finish first' : 'worker stopped');
  setWorkerState({ running: false, status: activeProcess ? 'stopping' : 'idle', message: activeProcess ? 'Worker stopping after current task...' : 'Worker stopped', nextSchedules: {} });
  return workerState;
}

function restartWorkerIfRunning() {
  if (!workerRunning) return;
  appendUiLog('worker settings changed; rescheduling tasks');
  clearTimers();
  startHeartbeat();
  scheduleAll();
}

function scheduleAll() {
  clearTimers();
  const settings = loadSettings();
  appendUiLog(`worker schedule loaded: stock_analysis=${settings.stockAnalysisEnabled ? `${settings.stockAnalysisPollSeconds}s` : 'disabled'}, daily_bundle=${settings.dailyBundleEnabled ? `${settings.dailyBundleCheckSeconds}s` : 'disabled'}, holding_prices=${settings.holdingPriceEnabled ? `${settings.holdingPricePollSeconds}s` : 'disabled'}`, 'debug');
  for (const [name, job] of Object.entries(AUTOMATIC_JOBS)) {
    if (settings[job.enabledKey] !== false) scheduleAutomaticJob(name, job, settings, true);
    else setNextSchedule(name, null);
  }
}

function scheduleAutomaticJob(name, job, settings, initial = false, overrideDelaySeconds = null) {
  if (!workerRunning || stopAfterCurrent) return;
  const nextRunAt = nextRunForJob(name, job, settings, initial, overrideDelaySeconds);
  if (!nextRunAt) {
    setNextSchedule(name, null);
    return;
  }
  setNextSchedule(name, nextRunAt.toISOString());
  appendUiLog(`${workerTaskLabel(name)} scheduled at ${formatDateTime(nextRunAt)}`, 'debug');
  const delayMs = Math.max(1, nextRunAt.getTime() - Date.now());
  const timer = setTimeout(() => runAutomaticJob(name, job, settings), delayMs);
  timers.push(timer);
}

async function runAutomaticJob(name, job, settings) {
    if (!workerRunning || stopAfterCurrent) return;
    if (activeProcess) {
      appendUiLog(`${workerTaskLabel(name)} delayed because another task is running`, 'debug');
      scheduleAutomaticJob(name, job, settings, false, Math.min(10, normalIntervalSeconds(job, settings)));
      return;
    }
    try {
      setNextSchedule(name, null);
      appendUiLog(`${workerTaskLabel(name)} started`, 'debug');
      const result = await job.run();
      if (!workerRunning || stopAfterCurrent) return;
      if (result && result.ok === false) {
        const text = [result.errorOutput, result.output].filter(Boolean).join('\n');
        const status = classifyError(text);
        const retryDelay = retryDelaySeconds(name, status);
        appendUiLog(`${workerTaskLabel(name)} failed: ${status}${retryDelay !== null ? `; retry in ${retryDelay}s` : ''}`, 'error');
        notifyAutomaticTask(name, 'error', status);
        setWorkerState({ taskType: name, status, lastError: text || 'Task failed', message: status });
        if (retryDelay === null && status !== 'local_data_missing') return;
        scheduleAutomaticJob(name, job, loadSettings(), false, retryDelay ?? normalIntervalSeconds(job, settings));
        return;
      }
      retryState.delete(name);
      markAutomaticJobSuccess(name, result);
      appendAutomaticResultLog(name, result);
      scheduleAutomaticJob(name, job, loadSettings(), false);
    } catch (error) {
      if (!workerRunning || stopAfterCurrent) return;
      const status = classifyError(error);
      const retryDelay = retryDelaySeconds(name, status);
      appendUiLog(`${workerTaskLabel(name)} failed: ${String(error)}${retryDelay !== null ? `; retry in ${retryDelay}s` : ''}`, 'error');
      notifyAutomaticTask(name, 'error', String(error));
      setWorkerState({ taskType: name, status, lastError: String(error), message: String(error) });
      if (retryDelay !== null || status === 'local_data_missing') scheduleAutomaticJob(name, job, loadSettings(), false, retryDelay ?? normalIntervalSeconds(job, settings));
    }
}

function nextRunForJob(name, job, settings, initial, overrideDelaySeconds) {
  if (overrideDelaySeconds !== null && overrideDelaySeconds !== undefined) {
    return new Date(Date.now() + Math.max(1, Number(overrideDelaySeconds)) * 1000);
  }
  if (job.mode === 'daily') return nextDailyRunAt(name, settings, initial);
  const delay = initial ? 1 : normalIntervalSeconds(job, settings);
  return new Date(Date.now() + delay * 1000);
}

function normalIntervalSeconds(job, settings) {
  return Math.max(1, Number(settings[job.intervalKey] || 60));
}

function nextDailyRunAt(name, settings, initial) {
  const today = localDateKey();
  const state = automaticJobState[name] || {};
  const windows = normalizeAllowedWindows(settings.dailyBundleAllowedWindows);
  if (state.lastSuccessDate === today) {
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    tomorrow.setHours(0, 0, 0, 0);
    return nextWindowStart(windows, new Date(tomorrow.getTime() - 1));
  }
  const now = new Date();
  if (isWithinAnyWindow(now, windows)) return new Date(Date.now() + (initial ? 1000 : normalIntervalSeconds(AUTOMATIC_JOBS[name], settings) * 1000));
  return nextWindowStart(windows, now);
}

function clearTimers() {
  for (const timer of timers) clearTimeout(timer);
  timers = [];
}

function startHeartbeat() {
  stopHeartbeat();
  sendHeartbeat('online', 'Worker scheduler running');
  const settings = loadSettings();
  const intervalMs = Math.max(5, Number(settings.workerStatusIntervalSeconds || 30)) * 1000;
  heartbeatTimer = setInterval(() => {
    if (workerRunning && !stopAfterCurrent) sendHeartbeat('online', 'Worker scheduler running');
  }, intervalMs);
}

function stopHeartbeat() {
  if (!heartbeatTimer) return;
  clearInterval(heartbeatTimer);
  heartbeatTimer = null;
}

async function sendHeartbeat(status, message) {
  const settings = loadSettings();
  const env = loadEnv(configPaths().env);
  const token = env[settings.workerTokenEnv] || process.env[settings.workerTokenEnv];
  const payload = {
    worker_id: settings.workerId,
    status,
    worker_version: workerVersion(),
    capabilities: workerCapabilities(settings),
    message: sanitizeHeartbeatMessage(message)
  };
  if (!token) {
    const error = `missing required environment variable: ${settings.workerTokenEnv}`;
    updateHeartbeatState('error', error);
    if (shouldEmitThrottledLog('worker_status:error', 60 * 1000)) appendUiLog(`Worker heartbeat failed: ${error}`, 'error');
    notifyAutomaticTask('worker_status', 'error', error);
    return;
  }
  try {
    const response = await postJson(buildApiUrl(settings.appBaseUrl, settings.workerStatusPath), payload, token);
    updateHeartbeatState(String(response.status || status), null);
    appendUiLog(`Worker heartbeat ${response.status || status}`, 'debug');
  } catch (error) {
    const messageText = sanitizeHeartbeatMessage(String(error));
    updateHeartbeatState('error', messageText);
    if (shouldEmitThrottledLog('worker_status:error', 60 * 1000)) appendUiLog(`Worker heartbeat failed: ${messageText}`, 'warn');
    notifyAutomaticTask('worker_status', 'error', messageText);
  }
}

function shouldEmitThrottledLog(key, throttleMs) {
  const now = Date.now();
  const stateKey = `log:${key}`;
  const last = notificationState.get(stateKey) || 0;
  if (now - last < throttleMs) return false;
  notificationState.set(stateKey, now);
  return true;
}

function postJson(url, payload, token) {
  return new Promise((resolve, reject) => {
    let parsed;
    try {
      parsed = new URL(url);
    } catch (error) {
      reject(new Error(`invalid worker status URL: ${url}`));
      return;
    }
    const data = JSON.stringify(payload);
    const client = parsed.protocol === 'https:' ? https : http;
    const request = client.request({
      protocol: parsed.protocol,
      hostname: parsed.hostname,
      port: parsed.port,
      path: `${parsed.pathname}${parsed.search}`,
      method: 'POST',
      timeout: 15000,
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(data)
      }
    }, (response) => {
      let body = '';
      response.setEncoding('utf8');
      response.on('data', (chunk) => { body += chunk; });
      response.on('end', () => {
        if (response.statusCode < 200 || response.statusCode >= 300) {
          reject(new Error(`HTTP ${response.statusCode}: ${body.slice(0, 300)}`));
          return;
        }
        try {
          resolve(body ? JSON.parse(body) : {});
        } catch (_) {
          resolve({});
        }
      });
    });
    request.on('timeout', () => {
      request.destroy(new Error('worker status request timed out'));
    });
    request.on('error', reject);
    request.write(data);
    request.end();
  });
}

function updateHeartbeatState(status, error) {
  setWorkerState({
    lastHeartbeatAt: new Date().toISOString(),
    lastHeartbeatStatus: status,
    lastHeartbeatError: error
  });
}

function workerVersion() {
  try {
    return app.getVersion();
  } catch (_) {
    return '0.1.0';
  }
}

function runAnalysisPoll() {
  const settings = loadSettings();
  appendUiLog(`Stock analysis queue target: ${settings.appBaseUrl}/api/worker/analysis-requests/claim`, 'debug');
  return pythonCommand(['app-worker', 'run-once', '--worker-config', configPaths().appWorker, '--config', settings.storagePath], 'command-log', 'stock_analysis');
}

function runHoldingRefresh() {
  const settings = loadSettings();
  appendUiLog(`Holding prices target: ${settings.appBaseUrl}/api/worker/holding-prices/watchlist`, 'debug');
  return pythonCommand(['app-worker', 'refresh-holding-prices', '--worker-config', configPaths().appWorker, '--config', settings.storagePath], 'command-log', 'holding_prices');
}

function runDailyCheck() {
  const settings = loadSettings();
  return pythonCommand(['app-worker', 'daily-check', '--worker-config', configPaths().appWorker, '--config', settings.storagePath, '--auto-pipeline', '--json-events'], 'command-log', 'daily_bundle');
}

function workerTaskLabel(taskName) {
  return {
    stock_analysis: 'Stock analysis queue',
    daily_bundle: 'Daily bundle',
    holding_prices: 'Holding prices',
    worker_status: 'Worker heartbeat'
  }[taskName] || taskName;
}

function classifyError(error) {
  const text = String(error);
  if (/401|403|token/i.test(text)) return 'auth_failed';
  if (/400|validation/i.test(text)) return 'validation_failed';
  if (/5\d\d|network/i.test(text)) return 'network_error';
  if (/Candidate 002 factor run|Tushare|no quote|empty watchlist|local data/i.test(text)) return 'local_data_missing';
  return 'failed';
}

function loadAutomaticJobState() {
  try {
    const statePath = path.join(configPaths().state, 'automatic-job-state.json');
    if (!fs.existsSync(statePath)) return {};
    return JSON.parse(fs.readFileSync(statePath, 'utf8'));
  } catch (_) {
    return {};
  }
}

function saveAutomaticJobState() {
  const paths = configPaths();
  fs.mkdirSync(paths.state, { recursive: true });
  fs.writeFileSync(path.join(paths.state, 'automatic-job-state.json'), JSON.stringify(automaticJobState, null, 2), 'utf8');
}

function markAutomaticJobSuccess(name, result) {
  if (AUTOMATIC_JOBS[name]?.mode !== 'daily') return;
  automaticJobState[name] = {
    lastSuccessDate: localDateKey(),
    lastSuccessAt: new Date().toISOString(),
    lastResultSummary: summarizeProcessResult(result)
  };
  saveAutomaticJobState();
}

function appendAutomaticResultLog(name, result) {
  const summary = summarizeProcessResult(result);
  if (AUTOMATIC_JOBS[name]?.mode === 'periodic' && /queue empty|watchlist empty/i.test(summary)) {
    appendUiLog(`${workerTaskLabel(name)} ok${summary ? `: ${summary}` : ''}`, 'debug');
    return;
  }
  appendUiLog(`${workerTaskLabel(name)} ok${summary ? `: ${summary}` : ''}`, 'info');
  notifyAutomaticTask(name, 'info', 'ok');
}

function summarizeProcessResult(result) {
  const text = [result?.output, result?.errorOutput].filter(Boolean).join('\n');
  if (!text) return '';
  const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const line = lines[index];
    if (!line.startsWith('{') && !line.startsWith('[event]')) return line.slice(0, 500);
  }
  return '';
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
    lastHeartbeatAt: null,
    lastHeartbeatStatus: 'unknown',
    lastHeartbeatError: null,
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
