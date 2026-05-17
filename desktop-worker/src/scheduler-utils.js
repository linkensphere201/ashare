const { normalizeAllowedWindows } = require('./worker-config');

function localDateKey(date = new Date()) {
  const pad = (value) => String(value).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

function parseTimeOfDay(value) {
  const normalized = String(value || '00:00').trim();
  if (normalized === '24:00') return 24 * 60;
  const [hourRaw, minuteRaw] = normalized.split(':');
  const hour = Math.max(0, Math.min(23, Number(hourRaw) || 0));
  const minute = Math.max(0, Math.min(59, Number(minuteRaw) || 0));
  return hour * 60 + minute;
}

function parseWindow(value) {
  const [startRaw, endRaw] = String(value).split('-');
  return { start: parseTimeOfDay(startRaw), end: parseTimeOfDay(endRaw || startRaw) };
}

function minutesSinceLocalMidnight(date) {
  return date.getHours() * 60 + date.getMinutes();
}

function isWithinAnyWindow(date, windows) {
  const minute = minutesSinceLocalMidnight(date);
  return normalizeAllowedWindows(windows).some((item) => {
    const window = parseWindow(item);
    if (window.start <= window.end) return minute >= window.start && minute < window.end;
    return minute >= window.start || minute < window.end;
  });
}

function nextWindowStart(windows, fromDate) {
  const candidates = [];
  for (let dayOffset = 0; dayOffset <= 2; dayOffset += 1) {
    const day = startOfLocalDay(addDays(fromDate, dayOffset));
    for (const item of normalizeAllowedWindows(windows)) {
      const window = parseWindow(item);
      const candidate = new Date(day.getTime() + window.start * 60 * 1000);
      if (candidate > fromDate) candidates.push(candidate);
    }
  }
  return candidates.sort((left, right) => left.getTime() - right.getTime())[0] || new Date(Date.now() + 60 * 1000);
}

function startOfLocalDay(date) {
  const output = new Date(date);
  output.setHours(0, 0, 0, 0);
  return output;
}

function addDays(date, days) {
  const output = new Date(date);
  output.setDate(output.getDate() + days);
  return output;
}

module.exports = {
  isWithinAnyWindow,
  localDateKey,
  nextWindowStart,
  parseTimeOfDay,
  parseWindow
};
