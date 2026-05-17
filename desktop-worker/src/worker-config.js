const DEFAULT_DAILY_WINDOWS = ['16:00-24:00', '00:00-10:00'];
const LOG_LEVELS = { debug: 10, info: 20, warn: 30, error: 40 };

function parseWorkerYaml(text) {
  const config = {};
  let section = '';
  let nested = '';
  let listKey = '';
  for (const rawLine of text.split(/\r?\n/)) {
    if (!rawLine.trim() || rawLine.trim().startsWith('#')) continue;
    const indent = rawLine.match(/^\s*/)[0].length;
    const line = rawLine.trim();
    if (line.startsWith('- ') && listKey) {
      const item = line.slice(2).trim().replace(/^["']|["']$/g, '');
      config[listKey] = [...(config[listKey] || []), item];
      continue;
    }
    listKey = '';
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
    if (!value && section === 'tasks' && nested && indent >= 4) {
      listKey = taskKey(nested, key.trim());
      config[listKey] = [];
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
    holding_prices: 'holding_prices_path',
    worker_status: 'worker_status_path'
  }[key] || key;
}

function taskKey(task, key) {
  const prefix = { stock_analysis: 'stock_analysis', daily_bundle: 'daily_bundle', holding_prices: 'holding_price' }[task] || task;
  if (key === 'enabled') return `${prefix}_enabled`;
  if (task === 'stock_analysis' && key === 'poll_interval_seconds') return 'poll_interval_seconds';
  if (task === 'daily_bundle' && key === 'check_interval_seconds') return 'daily_bundle_check_interval_seconds';
  if (task === 'daily_bundle' && key === 'earliest_publish_time') return 'earliest_daily_publish_time';
  if (task === 'daily_bundle' && key === 'allowed_windows') return 'daily_bundle_allowed_windows';
  if (task === 'holding_prices' && key === 'poll_interval_seconds') return 'holding_price_poll_interval_seconds';
  return `${prefix}_${key}`;
}

function normalizeLogLevel(value) {
  const level = String(value || 'info').trim().toLowerCase();
  return LOG_LEVELS[level] ? level : 'info';
}

function normalizeAllowedWindows(value) {
  const input = Array.isArray(value) ? value : String(value || '').split(',');
  const windows = input.map((item) => String(item || '').trim()).filter(Boolean);
  return windows.length ? windows : DEFAULT_DAILY_WINDOWS;
}

function mergeSettingsIntoWorkerYaml(text, settings, defaultStoragePath) {
  let lines = text.split(/\r?\n/);
  lines = setScalar(lines, [], 'app_base_url', settings.appBaseUrl || 'http://127.0.0.1:3000');
  lines = setScalar(lines, [], 'storage_config_path', settings.storagePath || defaultStoragePath);
  lines = setScalar(lines, [], 'log_level', normalizeLogLevel(settings.logLevel || 'info'));
  lines = setScalar(lines, [], 'poll_interval_seconds', Number(settings.stockAnalysisPollSeconds || 15));
  lines = setScalar(lines, [], 'holding_price_poll_interval_seconds', Number(settings.holdingPricePollSeconds || 300));
  lines = setScalar(lines, [], 'daily_bundle_check_interval_seconds', Number(settings.dailyBundleCheckSeconds || 900));
  lines = setScalar(lines, [], 'earliest_daily_publish_time', quote(settings.earliestDailyPublishTime || '16:30'));
  lines = setScalar(lines, ['tasks', 'stock_analysis'], 'enabled', settings.stockAnalysisEnabled !== false);
  lines = setScalar(lines, ['tasks', 'stock_analysis'], 'poll_interval_seconds', Number(settings.stockAnalysisPollSeconds || 15));
  lines = setScalar(lines, ['tasks', 'daily_bundle'], 'enabled', settings.dailyBundleEnabled !== false);
  lines = setScalar(lines, ['tasks', 'daily_bundle'], 'check_interval_seconds', Number(settings.dailyBundleCheckSeconds || 900));
  lines = setScalar(lines, ['tasks', 'daily_bundle'], 'earliest_publish_time', quote(settings.earliestDailyPublishTime || '16:30'));
  lines = setList(lines, ['tasks', 'daily_bundle'], 'allowed_windows', normalizeAllowedWindows(settings.dailyBundleAllowedWindows).map(quote));
  lines = setScalar(lines, ['tasks', 'holding_prices'], 'enabled', settings.holdingPriceEnabled !== false);
  lines = setScalar(lines, ['tasks', 'holding_prices'], 'poll_interval_seconds', Number(settings.holdingPricePollSeconds || 300));
  return `${trimTrailingBlankLines(lines).join('\n')}\n`;
}

function quote(value) {
  return `"${String(value).replace(/"/g, '\\"')}"`;
}

function trimTrailingBlankLines(lines) {
  const output = [...lines];
  while (output.length && output[output.length - 1] === '') output.pop();
  return output;
}

function setScalar(lines, path, key, value) {
  const rendered = `${indent(path.length)}${key}: ${value}`;
  const section = findSection(lines, path);
  if (!section) {
    const output = [...lines];
    ensureSection(output, path);
    const created = findSection(output, path);
    output.splice(created.end, 0, rendered);
    return output;
  }
  const index = findKey(lines, section.start + 1, section.end, path.length * 2, key);
  if (index !== -1) {
    const output = [...lines];
    output[index] = rendered;
    return output;
  }
  const output = [...lines];
  output.splice(section.end, 0, rendered);
  return output;
}

function setList(lines, path, key, values) {
  const section = findSection(lines, path);
  const replacement = [`${indent(path.length)}${key}:`, ...values.map((item) => `${indent(path.length + 1)}- ${item}`)];
  if (!section) {
    const output = [...lines];
    ensureSection(output, path);
    const created = findSection(output, path);
    output.splice(created.end, 0, ...replacement);
    return output;
  }
  const keyIndex = findKey(lines, section.start + 1, section.end, path.length * 2, key);
  const output = [...lines];
  if (keyIndex === -1) {
    output.splice(section.end, 0, ...replacement);
    return output;
  }
  let end = keyIndex + 1;
  while (end < section.end && lines[end].trim().startsWith('- ')) end += 1;
  output.splice(keyIndex, end - keyIndex, ...replacement);
  return output;
}

function ensureSection(lines, path) {
  for (let depth = 0; depth < path.length; depth += 1) {
    const prefix = path.slice(0, depth + 1);
    if (findSection(lines, prefix)) continue;
    const parent = findSection(lines, path.slice(0, depth));
    const insertAt = parent ? parent.end : lines.length;
    lines.splice(insertAt, 0, `${indent(depth)}${path[depth]}:`);
  }
}

function findSection(lines, path) {
  if (!path.length) return { start: -1, end: lines.length };
  let start = -1;
  let searchFrom = 0;
  for (let depth = 0; depth < path.length; depth += 1) {
    const key = path[depth];
    const level = depth * 2;
    start = -1;
    for (let index = searchFrom; index < lines.length; index += 1) {
      if (indentOf(lines[index]) === level && lines[index].trim() === `${key}:`) {
        start = index;
        searchFrom = index + 1;
        break;
      }
      if (depth > 0 && indentOf(lines[index]) <= level - 2 && lines[index].trim()) return null;
    }
    if (start === -1) return null;
  }
  const level = (path.length - 1) * 2;
  let end = lines.length;
  for (let index = start + 1; index < lines.length; index += 1) {
    if (lines[index].trim() && indentOf(lines[index]) <= level) {
      end = index;
      break;
    }
  }
  return { start, end };
}

function findKey(lines, start, end, level, key) {
  const pattern = new RegExp(`^\\s{${level}}${escapeRegExp(key)}\\s*:`);
  for (let index = start; index < end; index += 1) {
    if (pattern.test(lines[index])) return index;
  }
  return -1;
}

function indent(level) {
  return ' '.repeat(level * 2);
}

function indentOf(line) {
  return line.match(/^\s*/)[0].length;
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

module.exports = {
  DEFAULT_DAILY_WINDOWS,
  LOG_LEVELS,
  mergeSettingsIntoWorkerYaml,
  normalizeAllowedWindows,
  normalizeLogLevel,
  parseWorkerYaml
};
