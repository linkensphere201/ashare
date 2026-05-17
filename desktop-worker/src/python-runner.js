const DEFAULT_NO_PROXY = '127.0.0.1,localhost,101.34.212.101';

function runtimeEnv(localEnv, baseEnv = process.env) {
  const noProxy = mergeNoProxy(localEnv.NO_PROXY || baseEnv.NO_PROXY, localEnv.no_proxy || baseEnv.no_proxy);
  const merged = {
    ...baseEnv,
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

module.exports = {
  DEFAULT_NO_PROXY,
  mergeNoProxy,
  runtimeEnv
};
