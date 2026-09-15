import React, { useEffect, useState, useCallback } from 'react'

const api = async (path, body) => {
  const res = await fetch(`/api/${path}`, body
    ? { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body) }
    : undefined)
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText)
  return res.json()
}

const TIER_COLOR = {
  gpu: 'var(--ok)', int8: 'var(--warn)', cpu: 'var(--dim)',
  dropped: 'var(--bad)', crop: 'var(--dim)', recomputed: 'var(--bad)'
}


/* ------------------------------------------------------------------ */
/* Each forensic tool declares different parameters, and the server    */
/* validates them. Guessing one shape for all of them fails - and      */
/* hard-coding a place id breaks on every generated case, so the       */
/* scene comes from the backend.                                       */
/* ------------------------------------------------------------------ */
function argsFor (toolName, suspectId, suspect, state) {
  const name = toolName.replace(/_tool$/, '')
  const scene = state.case.scene || ''
  const at = (state.case.window && state.case.window.from) || '21:00'
  const lastAnswer = suspect.turns.length
    ? suspect.turns[suspect.turns.length - 1].answer
    : ''

  switch (name) {
    case 'check_alibi':
      return { suspect_id: suspectId, place: scene, at }
    case 'who_was_at':
      return { place: scene, at }
    case 'movements_of':
      return { suspect_id: suspectId }
    case 'verify_statement':
      return { suspect_id: suspectId, statement: lastAnswer }
    case 'lookup_evidence':
      return { query: scene }
    default:
      return { suspect_id: suspectId }
  }
}

/* ------------------------------------------------------------------ */
/* Suspect cards: composure is a real state variable, so it gets a bar */
/* ------------------------------------------------------------------ */
function SuspectCard ({ s, active, onClick }) {
  const pct = Math.round(s.composure * 100)
  const tone = s.composure > 0.6 ? 'ok' : s.composure > 0.35 ? 'warn' : 'bad'
  return (
    <button className={`card ${active ? 'active' : ''}`} onClick={onClick}>
      <div className="card-name">{s.name}</div>
      <div className="card-role">{s.role}</div>
      <div className="bar"><div className={`fill ${tone}`} style={{ width: `${pct}%` }} /></div>
      <div className="card-meta">
        composure {s.composure.toFixed(2)} · temp {s.temperature}
      </div>
      <div className="card-meta dim">
        {s.cache_tokens} tok cached · {s.turns.length} turns
        {s.hits.length > 0 && <span className="flag"> {s.hits.length} caught</span>}
      </div>
    </button>
  )
}

/* ------------------------------------------------------------------ */
/* Branch tree: the whole point of the right panel. Committed turns    */
/* run down the trunk; the last explore fans out into three forks.     */
/* ------------------------------------------------------------------ */
function BranchTree ({ suspect, candidates, chosen }) {
  const turns = suspect ? suspect.turns : []
  const rowH = 30
  const trunkX = 26
  const height = Math.max(120, turns.length * rowH + 110)

  return (
    <svg viewBox={`0 0 260 ${height}`} className="tree">
      {turns.map((t, i) => {
        const y = 18 + i * rowH
        return (
          <g key={i}>
            {i > 0 && <line x1={trunkX} y1={y - rowH} x2={trunkX} y2={y}
                            stroke="var(--line)" strokeWidth="1.5" />}
            <circle cx={trunkX} cy={y} r="5"
                    fill={TIER_COLOR[t.snapshot_tier] || 'var(--dim)'} />
            <text x={trunkX + 12} y={y + 4} className="lbl">
              t{t.index} · {t.snapshot_tier}
            </text>
          </g>
        )
      })}

      {candidates && candidates.length > 0 && (() => {
        const baseY = 18 + Math.max(0, turns.length - 1) * rowH
        const forkY = baseY + rowH
        return (
          <g>
            <line x1={trunkX} y1={baseY} x2={trunkX} y2={forkY}
                  stroke="var(--line)" strokeWidth="1.5" />
            <circle cx={trunkX} cy={forkY} r="6" fill="var(--accent)" />
            <text x={trunkX + 12} y={forkY + 4} className="lbl">fork point</text>
            {candidates.map((c, i) => {
              const y = forkY + 26 + i * 26
              const picked = chosen === i
              return (
                <g key={i}>
                  <line x1={trunkX} y1={forkY} x2={86} y2={y}
                        stroke={picked ? 'var(--ok)' : 'var(--line)'}
                        strokeWidth={picked ? 2 : 1.2} />
                  <circle cx="86" cy={y} r="4.5"
                          fill={picked ? 'var(--ok)' : 'var(--dim)'} />
                  <text x="96" y={y + 4} className="lbl">
                    {c.band} {c.score.toFixed(2)}
                  </text>
                </g>
              )
            })}
          </g>
        )
      })()}
    </svg>
  )
}

/* ------------------------------------------------------------------ */
function CachePanel ({ state }) {
  if (!state) return null
  const a = state.accounting
  const st = state.store
  const pct = st.budget_bytes
    ? Math.min(100, Math.round(100 * st.gpu_bytes / st.budget_bytes)) : 0

  return (
    <>
      <div className="grid2">
        <div className="stat">
          <div className="stat-label">prefill tokens</div>
          <div className="stat-value">{a.prefill_tokens.toLocaleString()}</div>
        </div>
        <div className="stat">
          <div className="stat-label">saved on shared block</div>
          <div className="stat-value">{a.tokens_saved_on_shared_block.toLocaleString()}</div>
        </div>
        <div className="stat">
          <div className="stat-label">forks / crops</div>
          <div className="stat-value">{a.forks} / {a.crops}</div>
        </div>
        <div className="stat">
          <div className="stat-label">evictions</div>
          <div className="stat-value">{a.evictions}</div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">snapshot store</div>
        <div className="bar tall"><div className="fill ok" style={{ width: `${pct}%` }} /></div>
        <div className="mono small">{st.report}</div>
        <div className="mono small dim">
          hits gpu {st.stats.gpu} · int8 {st.stats.int8} · cpu {st.stats.cpu}
          {' '}· misses {st.stats.misses} · hit rate {Math.round(st.stats.hit_rate * 100)}%
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">cache log</div>
        <div className="log">
          {state.log.slice().reverse().map((e, i) => (
            <div key={i} className="mono small">
              <span className={`kind ${e.kind}`}>{e.kind}</span>
              <span className="dim"> {String(e.tokens).padStart(5)} </span>
              {e.label}
            </div>
          ))}
        </div>
      </div>
    </>
  )
}

/* ------------------------------------------------------------------ */
export default function App () {
  const [state, setState] = useState(null)
  const [sel, setSel] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)
  const [typed, setTyped] = useState('')
  const [chosen, setChosen] = useState(null)
  const [verdict, setVerdict] = useState(null)
  const [tools, setTools] = useState([])
  const [toolOut, setToolOut] = useState(null)

  const refresh = useCallback(async () => {
    const s = await api('state')
    setState(s)
    if (!sel && s.suspects.length) setSel(s.suspects[0].id)
    return s
  }, [sel])

  useEffect(() => { refresh().catch(e => setErr(String(e))) }, [])
  useEffect(() => {
    api('tools').then(d => setTools(d.tools)).catch(() => {})
  }, [])

  const run = async (fn) => {
    setBusy(true); setErr(null)
    try { await fn(); await refresh() }
    catch (e) { setErr(String(e.message || e)) }
    finally { setBusy(false) }
  }

  if (!state) {
    return <div className="boot">{err ? `error: ${err}` : 'loading model…'}</div>
  }

  const suspect = state.suspects.find(s => s.id === sel)
  const candidates = state.candidates[sel] || []

  return (
    <div className="app">
      <header>
        <h1>{state.case.title}</h1>
        <div className="sub">
          {state.accounting.shared_tokens} shared tokens forked to{' '}
          {state.accounting.suspects} suspects ·{' '}
          {state.accounting.bytes_per_token.toLocaleString()} B/token
        </div>
      </header>

      <div className="cols">
        {/* ------------------------------- left: the game */}
        <section className="left">
          <div className="cards">
            {state.suspects.map(s => (
              <SuspectCard key={s.id} s={s} active={s.id === sel}
                           onClick={() => { setSel(s.id); setChosen(null) }} />
            ))}
          </div>

          <div className="transcript">
            {suspect.turns.length === 0 && (
              <div className="dim small">No questions yet.</div>
            )}
            {suspect.turns.map(t => (
              <div key={t.index} className="turn">
                <div className="q">You: {t.question}</div>
                <div className="a">{suspect.name}: {t.answer}</div>
                {t.granted.length > 0 && (
                  <div className="note granted">
                    learned: {t.granted.join(', ')}
                  </div>
                )}
                <button className="mini"
                        disabled={busy}
                        onClick={() => run(() =>
                          api('rewind', { suspect_id: sel, turn_index: t.index }))}>
                  rewind to here
                </button>
              </div>
            ))}
            {suspect.hits.map((h, i) => (
              <div key={`h${i}`} className="note bad">contradiction: {h.note}</div>
            ))}
          </div>

          {candidates.length > 0 ? (
            <div className="options">
              <div className="panel-title">three forks explored — pick one</div>
              {candidates.map(c => (
                <button key={c.index} disabled={busy}
                        className={`option ${chosen === c.index ? 'picked' : ''}`}
                        onClick={() => { setChosen(c.index) }}>
                  <div className="option-head">
                    <span className={`band ${c.band}`}>{c.band}</span>
                    <span className="mono small dim">{c.score.toFixed(2)}</span>
                    <span className="q">{c.question}</span>
                  </div>
                  <div className="preview dim small">{c.preview}</div>
                  {c.contradiction ? (
                    <div className="note bad small">{c.contradiction_note}</div>
                  ) : null}
                </button>
              ))}
              <button className="primary" disabled={busy || chosen === null}
                      onClick={() => run(async () => {
                        await api('commit', { suspect_id: sel, index: chosen })
                        setChosen(null)
                      })}>
                commit this line
              </button>
            </div>
          ) : (
            <button className="primary" disabled={busy}
                    onClick={() => run(() => api('explore', { suspect_id: sel }))}>
              explore three questions
            </button>
          )}

          <div className="row">
            <input value={typed} placeholder="or ask your own"
                   onChange={e => setTyped(e.target.value)}
                   onKeyDown={e => {
                     if (e.key === 'Enter' && typed.trim()) {
                       run(async () => {
                         await api('ask', { suspect_id: sel, question: typed })
                         setTyped('')
                       })
                     }
                   }} />
            <button disabled={busy || !typed.trim()}
                    onClick={() => run(async () => {
                      await api('ask', { suspect_id: sel, question: typed })
                      setTyped('')
                    })}>ask</button>
          </div>

          <div className="row wrap">
            {state.suspects.filter(s => s.id !== sel && s.turns.length > 0).map(src => (
              <button key={src.id} disabled={busy}
                      onClick={() => run(() => api('quote', {
                        from_suspect: src.id,
                        turn_index: src.turns.length - 1,
                        to_suspect: sel
                      }))}>
                quote {src.name.split(' ').slice(-1)[0]} at them
              </button>
            ))}
            {state.suspects.map(s => (
              <button key={`acc-${s.id}`} className="danger" disabled={busy}
                      onClick={() => run(async () => {
                        setVerdict(await api('accuse', { suspect_id: s.id }))
                      })}>
                accuse {s.name.split(' ').slice(-1)[0]}
              </button>
            ))}
          </div>

          {tools.length > 0 && (
            <div className="row wrap">
              <span className="dim small">forensics (MCP):</span>
              {tools.map(t => (
                <button key={t.name} className="mini" disabled={busy}
                        onClick={() => run(async () => {
                          const r = await api('tool', {
                            name: t.name,
                            args: argsFor(t.name, sel, suspect, state)
                          })
                          setToolOut(r.result)
                        })}>
                  {t.name.replace('_tool', '')}
                </button>
              ))}
            </div>
          )}
          {toolOut && (
            <pre className="mono small toolout">{JSON.stringify(toolOut, null, 2)}</pre>
          )}

          {verdict && (
            <div className={`verdict ${verdict.correct ? 'ok' : 'bad'}`}>
              <div>You accused {verdict.accused_name}.</div>
              <div>The culprit was {verdict.culprit_name}.</div>
              <div>{verdict.correct ? 'CORRECT' : 'WRONG'} —{' '}
                {verdict.contradictions_found} contradictions surfaced.</div>
            </div>
          )}
          {err && <div className="note bad">{err}</div>}
        </section>

        {/* ------------------------------- right: the machinery */}
        <aside className="right">
          <div className="panel">
            <div className="panel-title">branch tree — {suspect.name}</div>
            <BranchTree suspect={suspect} candidates={candidates} chosen={chosen} />
          </div>
          <CachePanel state={state} />
          <button className="mini" disabled={busy}
                  onClick={() => run(async () => {
                    await api('reset', {}); setVerdict(null); setChosen(null)
                  })}>reset case</button>
        </aside>
      </div>
    </div>
  )
}