// One place that talks to the backend. Every call goes through here so
// error handling is uniform and the components stay small.

export async function api (path, body) {
  const res = await fetch(`/api/${path}`, body === undefined
    ? undefined
    : { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body) })
  if (!res.ok) {
    let detail = res.statusText
    try { detail = (await res.json()).detail || detail } catch {}
    throw new Error(detail)
  }
  return res.json()
}

// Poll a background job (generation, judging) until it settles.
export async function waitJob (jobId, onTick, everyMs = 1500) {
  for (;;) {
    const j = await api(`job/${jobId}`)
    if (onTick) onTick(j)
    if (j.status !== 'running') return j
    await new Promise(r => setTimeout(r, everyMs))
  }
}

export const last = (s) => s.split(' ').slice(-1)[0]