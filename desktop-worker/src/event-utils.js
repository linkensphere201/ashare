function formatWorkflowEvent(event, formattedDateTime) {
  const task = event.task_type || event.workflow_id || 'workflow';
  const step = event.step ? ` ${event.step}` : '';
  const progress = event.step_index && event.step_total ? ` ${event.step_index}/${event.step_total}` : '';
  const status = event.status || event.event || 'event';
  const message = event.message || '';
  return `[${formattedDateTime}] ${task}${step}${progress} ${status}: ${message}\n`;
}

function eventLogLevel(event) {
  const status = String(event.status || event.event || '').toLowerCase();
  if (status.includes('failed')) return 'error';
  if (status.includes('retry')) return 'warn';
  return 'info';
}

function normalizeUiEvent(event, context, automaticJobs, manualStages) {
  if (!context?.manualTaskType) {
    const taskType = event.task_type || event.workflow_id || context?.taskType;
    const status = String(event.status || event.event || '').toLowerCase();
    if (automaticJobs[taskType]?.mode === 'periodic' && !status.includes('failed')) {
      return { ...event, ui_log: false, ui_notify: false };
    }
    return event;
  }
  const mapped = mapManualStage(context.manualTaskType, event, manualStages);
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

function mapManualStage(taskType, event, manualStages) {
  const step = String(event.step || '');
  const status = String(event.status || event.event || '');
  if (taskType === 'sync') {
    if (step === 'upload_bundle') return manualStage(taskType, 'upload_bundle', event, manualStages);
    if (step === 'build_bundle') return manualStage(taskType, 'build_daily_bundle', event, manualStages);
    if (step === 'sync_latest' || step === 'sync_dry_run' || /sync/.test(step)) return manualStage(taskType, 'sync_latest', event, manualStages);
    if (step === 'quality_check') return manualStage(taskType, 'quality_check', event, manualStages);
    if (step === 'compute_candidate_002_factors') return manualStage(taskType, 'compute_candidate_002_factors', event, manualStages);
    if (step === 'rank_candidate_002' || step === 'backtest_candidate_002' || step === 'create_snapshot') return manualStage(taskType, 'rank_candidate_002', event, manualStages);
    if (step === 'build_daily_bundle' || /build/.test(step)) return manualStage(taskType, 'build_daily_bundle', event, manualStages);
    if (/failed|completed/.test(status)) return terminalManualStage(taskType, event, manualStages);
    return null;
  }
  if (taskType === 'stock_analysis') {
    if (step === 'validate_symbol') return manualStage(taskType, 'validate_symbol', event, manualStages);
    if (step === 'resolve_factor_run') return manualStage(taskType, 'resolve_factor_run', event, manualStages);
    if (step === 'analyze_stock') return manualStage(taskType, 'analyze_stock', event, manualStages);
    if (step === 'write_summary' || /completed|failed/.test(status)) return manualStage(taskType, 'write_summary', event, manualStages);
    return null;
  }
  if (taskType === 'holding_analysis') {
    if (step === 'load_watchlist') return manualStage(taskType, 'load_watchlist', event, manualStages);
    if (step === 'refresh_quotes') return manualStage(taskType, 'refresh_quotes', event, manualStages);
    if (step === 'upload_prices' || /completed|failed/.test(status)) return manualStage(taskType, 'upload_prices', event, manualStages);
    return null;
  }
  return null;
}

function manualStage(taskType, stageKey, event = {}, manualStages = {}) {
  const stages = manualStages[taskType] || [];
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

function terminalManualStage(taskType, event = {}, manualStages = {}) {
  const stages = manualStages[taskType] || [];
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

function manualStageTotal(taskType, manualStages = {}) {
  return (manualStages[taskType] || []).length;
}

function isJsonOnlyChunk(text) {
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  return lines.length > 0 && lines.every((line) => line.trim().startsWith('{'));
}

module.exports = {
  eventLogLevel,
  formatWorkflowEvent,
  isJsonOnlyChunk,
  manualStage,
  manualStageTotal,
  normalizeUiEvent
};
