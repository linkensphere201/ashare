const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('stockPicker', {
  runCommand: (args, manualTaskType) => ipcRenderer.invoke('run-command', args, manualTaskType),
  stopCommand: () => ipcRenderer.invoke('stop-command'),
  startWorker: () => ipcRenderer.invoke('start-worker'),
  stopWorker: () => ipcRenderer.invoke('stop-worker'),
  getWorkerStatus: () => ipcRenderer.invoke('get-worker-status'),
  getManualTaskStatus: () => ipcRenderer.invoke('get-manual-task-status'),
  getRecentLogs: () => ipcRenderer.invoke('get-recent-logs'),
  getSettings: () => ipcRenderer.invoke('get-settings'),
  saveSettings: (settings) => ipcRenderer.invoke('save-settings', settings),
  notify: (title, body) => ipcRenderer.invoke('show-notification', title, body),
  onCommandLog: (callback) => ipcRenderer.on('command-log', (_event, payload) => callback(payload)),
  onWorkflowEvent: (callback) => ipcRenderer.on('workflow-event', (_event, payload) => callback(payload)),
  onWorkerStatus: (callback) => ipcRenderer.on('worker-status', (_event, payload) => callback(payload)),
  onManualTaskStatus: (callback) => ipcRenderer.on('manual-task-status', (_event, payload) => callback(payload))
});
