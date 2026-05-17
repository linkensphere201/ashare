const assert = require('node:assert/strict');
const test = require('node:test');
const { formatWorkflowEvent, normalizeUiEvent } = require('../src/event-utils');
const { sanitizeHeartbeatMessage } = require('../src/heartbeat-utils');
const { runtimeEnv } = require('../src/python-runner');
const { isWithinAnyWindow, nextWindowStart } = require('../src/scheduler-utils');
const { mergeSettingsIntoWorkerYaml, parseWorkerYaml } = require('../src/worker-config');

test('mergeSettingsIntoWorkerYaml preserves advanced worker settings', () => {
  const original = [
    'app_base_url: http://old.local',
    'storage_config_path: C:\\old\\storage.yaml',
    'worker_id: custom-worker',
    'worker_token_env: CUSTOM_WORKER_TOKEN',
    'tushare_token_env: CUSTOM_TUSHARE_TOKEN',
    'unknown_top_level: keep-me',
    'log_level: info',
    'api_paths:',
    '  analysis_claim: /custom/claim',
    '  analysis_result: /custom/result/{requestId}',
    '  worker_status: /custom/status',
    'tasks:',
    '  stock_analysis:',
    '    enabled: true',
    '    poll_interval_seconds: 15',
    '  daily_bundle:',
    '    enabled: true',
    '    check_interval_seconds: 900',
    '    earliest_publish_time: "16:30"',
    '    allowed_windows:',
    '      - "16:00-24:00"',
    '    upload_archive_max: 99',
    '  holding_prices:',
    '    enabled: true',
    '    poll_interval_seconds: 300',
    ''
  ].join('\n');

  const merged = mergeSettingsIntoWorkerYaml(original, {
    appBaseUrl: 'https://app.example',
    storagePath: 'D:\\stock\\storage.yaml',
    stockAnalysisEnabled: false,
    dailyBundleEnabled: true,
    holdingPriceEnabled: false,
    stockAnalysisPollSeconds: 31,
    dailyBundleCheckSeconds: 601,
    holdingPricePollSeconds: 401,
    earliestDailyPublishTime: '17:10',
    dailyBundleAllowedWindows: ['17:00-23:00', '00:00-09:00'],
    logLevel: 'debug'
  }, 'fallback-storage.yaml');

  assert.match(merged, /worker_id: custom-worker/);
  assert.match(merged, /worker_token_env: CUSTOM_WORKER_TOKEN/);
  assert.match(merged, /tushare_token_env: CUSTOM_TUSHARE_TOKEN/);
  assert.match(merged, /unknown_top_level: keep-me/);
  assert.match(merged, /analysis_claim: \/custom\/claim/);
  assert.match(merged, /worker_status: \/custom\/status/);
  assert.match(merged, /upload_archive_max: 99/);

  const parsed = parseWorkerYaml(merged);
  assert.equal(parsed.app_base_url, 'https://app.example');
  assert.equal(parsed.storage_config_path, 'D:\\stock\\storage.yaml');
  assert.equal(parsed.stock_analysis_enabled, false);
  assert.equal(parsed.daily_bundle_enabled, true);
  assert.equal(parsed.holding_price_enabled, false);
  assert.equal(parsed.poll_interval_seconds, 31);
  assert.equal(parsed.daily_bundle_check_interval_seconds, 601);
  assert.equal(parsed.holding_price_poll_interval_seconds, 401);
  assert.deepEqual(parsed.daily_bundle_allowed_windows, ['17:00-23:00', '00:00-09:00']);
});

test('scheduler window helpers handle same-day and next-day windows', () => {
  assert.equal(isWithinAnyWindow(new Date(2026, 4, 17, 17, 0), ['16:00-24:00']), true);
  assert.equal(isWithinAnyWindow(new Date(2026, 4, 17, 9, 30), ['16:00-24:00', '00:00-10:00']), true);
  assert.equal(isWithinAnyWindow(new Date(2026, 4, 17, 12, 0), ['16:00-24:00', '00:00-10:00']), false);

  const next = nextWindowStart(['16:00-24:00', '00:00-10:00'], new Date(2026, 4, 17, 12, 0));
  assert.equal(next.getFullYear(), 2026);
  assert.equal(next.getMonth(), 4);
  assert.equal(next.getDate(), 17);
  assert.equal(next.getHours(), 16);
  assert.equal(next.getMinutes(), 0);
});

test('heartbeat message sanitizer removes local paths and bearer tokens', () => {
  const sanitized = sanitizeHeartbeatMessage('failed at C:\\secret\\worker\\file.txt with Bearer abc.def-ghi');
  assert.equal(sanitized, 'failed at [local-path] with Bearer [token]');
});

test('event utils normalize periodic and manual progress events', () => {
  assert.deepEqual(
    normalizeUiEvent({ task_type: 'holding_prices', status: 'completed' }, { taskType: 'holding_prices' }, { holding_prices: { mode: 'periodic' } }, {}),
    { task_type: 'holding_prices', status: 'completed', ui_log: false, ui_notify: false }
  );
  const context = { manualTaskType: 'stock_analysis', lastStageKey: null, lastProgressLogAt: 0, lastProgressMessage: '' };
  const normalized = normalizeUiEvent(
    { step: 'analyze_stock', status: 'running', message: 'analyzing 600519.SH' },
    context,
    {},
    { stock_analysis: [['validate_symbol', 'Validate'], ['analyze_stock', 'Analyze']] }
  );
  assert.equal(normalized.task_type, 'stock_analysis');
  assert.equal(normalized.step, 'analyze_stock');
  assert.equal(normalized.step_index, 2);
  assert.equal(normalized.ui_log, true);
  assert.equal(formatWorkflowEvent(normalized, '2026-05-17 16:30:00'), '[2026-05-17 16:30:00] stock_analysis analyze_stock 2/2 running: analyzing 600519.SH\n');
});

test('python runtime env disables proxies and keeps no_proxy defaults', () => {
  const env = runtimeEnv({ NO_PROXY: 'example.local', HTTP_PROXY: 'http://proxy' }, { HTTPS_PROXY: 'http://proxy2' });
  assert.equal(env.PYTHONUNBUFFERED, '1');
  assert.equal(env.HTTP_PROXY, undefined);
  assert.equal(env.HTTPS_PROXY, undefined);
  assert.match(env.NO_PROXY, /example\.local/);
  assert.match(env.NO_PROXY, /127\.0\.0\.1/);
});
