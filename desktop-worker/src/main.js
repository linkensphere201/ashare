const { app, BrowserWindow, Menu, Notification, Tray, ipcMain, nativeImage } = require('electron');
const path = require('node:path');
const { spawn } = require('node:child_process');

let mainWindow;
let tray;
let activeProcess = null;

const repoRoot = path.resolve(__dirname, '..', '..');

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
    }
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
  const trayIconPath = path.join(__dirname, 'renderer', 'tray.ico');
  const image = require('node:fs').existsSync(trayIconPath)
    ? trayIconPath
    : nativeImage.createFromDataURL('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAGklEQVR4nGNkYGD4z0AEYBxVSF+FAgMDAH5pAhEMZnoXAAAAAElFTkSuQmCC');
  tray = new Tray(image);
  tray.setToolTip('Stock Picker Console');
  tray.setContextMenu(Menu.buildFromTemplate([
    { label: 'Open Console', click: () => showWindow() },
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

function pythonCommand(args, eventChannel) {
  if (activeProcess) {
    return Promise.reject(new Error('another workflow is already running'));
  }
  return new Promise((resolve) => {
    const executable = process.platform === 'win32'
      ? path.join(repoRoot, '.venv', 'Scripts', 'python.exe')
      : path.join(repoRoot, '.venv', 'bin', 'python');
    const fallback = process.platform === 'win32' ? 'python' : 'python3';
    const command = require('node:fs').existsSync(executable) ? executable : fallback;
    activeProcess = spawn(command, ['-m', 'stock_picker.cli', ...args], {
      cwd: repoRoot,
      env: process.env,
      windowsHide: true
    });
    let output = '';
    let errorOutput = '';
    activeProcess.stdout.on('data', (chunk) => {
      const text = chunk.toString();
      output += text;
      mainWindow?.webContents.send(eventChannel, { stream: 'stdout', text });
      for (const line of text.split(/\r?\n/)) {
        if (!line.trim().startsWith('{')) continue;
        try {
          const parsed = JSON.parse(line);
          mainWindow?.webContents.send('workflow-event', parsed);
        } catch (_) {
          // Keep non-event lines in the plain log.
        }
      }
    });
    activeProcess.stderr.on('data', (chunk) => {
      const text = chunk.toString();
      errorOutput += text;
      mainWindow?.webContents.send(eventChannel, { stream: 'stderr', text });
    });
    activeProcess.on('close', (code) => {
      activeProcess = null;
      const ok = code === 0;
      if (ok && Notification.isSupported()) {
        new Notification({ title: 'Stock Picker', body: 'Workflow completed' }).show();
      }
      resolve({ ok, code, output, errorOutput });
    });
  });
}

function stopActiveProcess() {
  if (!activeProcess) return false;
  activeProcess.kill();
  activeProcess = null;
  return true;
}

ipcMain.handle('run-command', async (_event, args) => pythonCommand(args, 'command-log'));
ipcMain.handle('stop-command', async () => stopActiveProcess());
ipcMain.handle('show-notification', async (_event, title, body) => {
  if (Notification.isSupported()) new Notification({ title, body }).show();
  return true;
});

app.whenReady().then(() => {
  createWindow();
  createTray();
  app.on('activate', showWindow);
});

app.on('window-all-closed', (event) => {
  event.preventDefault();
});
