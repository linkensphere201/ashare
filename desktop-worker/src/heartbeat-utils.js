function buildApiUrl(baseUrl, apiPath) {
  const base = String(baseUrl || '').replace(/\/+$/, '');
  const suffix = String(apiPath || '').startsWith('/') ? apiPath : `/${apiPath || ''}`;
  return `${base}${suffix}`;
}

function sanitizeHeartbeatMessage(message) {
  return String(message || '')
    .replace(/[A-Za-z]:\\[^\s"'<>]+/g, '[local-path]')
    .replace(/Bearer\s+[A-Za-z0-9._~+/-]+/gi, 'Bearer [token]')
    .slice(0, 180);
}

function workerCapabilities(settings) {
  return {
    daily_bundle: Boolean(settings.dailyBundleEnabled),
    stock_analysis: Boolean(settings.stockAnalysisEnabled),
    holding_prices: Boolean(settings.holdingPriceEnabled)
  };
}

module.exports = {
  buildApiUrl,
  sanitizeHeartbeatMessage,
  workerCapabilities
};
