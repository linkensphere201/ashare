const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('stockPicker', {
  runCommand: (args) => ipcRenderer.invoke('run-command', args),
  stopCommand: () => ipcRenderer.invoke('stop-command'),
  notify: (title, body) => ipcRenderer.invoke('show-notification', title, body),
  onCommandLog: (callback) => ipcRenderer.on('command-log', (_event, payload) => callback(payload)),
  onWorkflowEvent: (callback) => ipcRenderer.on('workflow-event', (_event, payload) => callback(payload))
});
