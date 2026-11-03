// data.js — load the static JSON exported by `python -m aml_gnn export-web`.
// Each file is fetched once and cached for the rest of the session.

const cache = new Map();

export function load(path) {
  if (!cache.has(path)) {
    const request = fetch(`data/${path}`)
      .then((response) => {
        if (!response.ok) throw new Error(`Could not load ${path} (HTTP ${response.status})`);
        return response.json();
      })
      .catch((error) => {
        cache.delete(path);            // allow a retry on the next visit
        throw error;
      });
    cache.set(path, request);
  }
  return cache.get(path);
}
