import React, { useEffect, useState } from 'react'
import { api, waitJob, last } from './api.js'
import { humanDetail } from './Labresult.jsx'

/* ================================================================== */
/*  Modal shell                                                        */
/* ================================================================== */
export function Modal ({ title, onClose, children, wide }) {
  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className={`modal ${wide ? 'wide' : ''}`} onClick={e => e.stopPropagation()}>
        <div className="modal-head">
          <span className="modal-title">{title}</span>
          <button className="x" onClick={onClose} aria-label="close">✕</button>
        </div>
        <div className="modal-body">{children}</div>
      </div>
    </div>
  )
}

/* ================================================================== */
/*  The case file - what the detective is allowed to read              */
/* ================================================================== */
export function CaseFile ({ file, onClose }) {
  const byType = {}
  for (const k of file.known) (byType[k.type] ||= []).push(k)
  return (
    <Modal title={`Case file — ${file.title}`} onClose={onClose} wide>
      <div className="stamp">{file.generated_by}</div>
      <p className="brief">{file.brief}</p>
      {file.scene && (
        <p className="dim small">
          Scene: <b>{file.scene.replace('loc_', '')}</b>
          {file.window?.from && <> · window <b>{file.window.from}–{file.window.to}</b></>}
        </p>
      )}

      <h4>Persons of interest</h4>
      <ul className="plain">
        {file.suspects.map(s => <li key={s.id}><b>{s.name}</b> — {s.role}</li>)}
      </ul>

      <h4>Known at the outset</h4>
      {Object.entries(byType).map(([t, items]) => (
        <div key={t}>
          <div className="dim small caps">{t}</div>
          <ul className="plain">{items.map(k => <li key={k.id}>{k.text}</li>)}</ul>
        </div>
      ))}

      <h4>Evidence board <span className="dim small">({file.board.length} pinned)</span></h4>
      {file.board.length === 0
        ? <p className="dim small">Nothing yet. Question people, quote them at each other, use the lab.</p>
        : <ul className="board">
            {file.board.map((b, i) => (
              <li key={i} className={`pin ${b.kind}`}>
                <span className="pin-kind">{b.kind}</span>
                <span>{b.text}</span>
                <span className="dim small"> — {b.via}</span>
              </li>
            ))}
          </ul>}
    </Modal>
  )
}

/* ================================================================== */
/*  Suspect dossier                                                     */
/* ================================================================== */
export function Dossier ({ s, onClose, onRewind, busy, hints, state }) {
  const tone = s.composure > 0.6 ? 'ok' : s.composure > 0.35 ? 'warn' : 'bad'
  return (
    <Modal title={`Dossier — ${s.name}`} onClose={onClose} wide>
      <div className="dossier-head">
        <div>
          <div className="dim">{s.role}</div>
          <div className="bar wide"><div className={`fill ${tone}`} style={{ width: `${Math.round(s.composure * 100)}%` }} /></div>
          <div className="small dim">composure {s.composure.toFixed(2)} · {s.turns.length} turns · {s.hits.length} caught</div>
        </div>
      </div>

      <h4>Pinned</h4>
      {s.pins.length === 0
        ? <p className="dim small">Nothing pinned yet.</p>
        : <ul className="board">
            {s.pins.map((p, i) => (
              <li key={i} className={`pin ${p.kind}`}>
                <span className="pin-kind">{p.kind}</span>
                <span>{humanDetail(p.text, state)}</span>
                {p.turn != null && <span className="dim small"> — turn {p.turn}</span>}
              </li>
            ))}
          </ul>}

      <h4>Statements</h4>
      {s.turns.length === 0 && <p className="dim small">Not yet questioned.</p>}
      {s.turns.map(t => (
        <div key={t.index} className="turn">
          <div className="q">
            {hints && t.kind && <span className="chip">{t.kind.replace('_', ' ')}</span>}
            You: {t.question}
          </div>
          <div className="a">{s.name}: {t.answer}</div>
          {t.band && <span className={`band ${t.band}`}>{t.band} · {t.score.toFixed(2)}</span>}
          {t.granted.length > 0 && (
            <div className="note granted">now knows: {t.granted.map(g => g.text).join(' ')}</div>
          )}
          <button className="mini" disabled={busy} onClick={() => onRewind(t.index)}>
            rewind to before this
          </button>
        </div>
      ))}
    </Modal>
  )
}

/* ================================================================== */
/*  Verdict                                                             */
/* ================================================================== */
export function Verdict ({ v, judging, onNewCase, onClose, state }) {
  const [tab, setTab] = useState('debrief')

  if (judging) {
    return (
      <Modal title="The judge is reading the transcript" onClose={() => {}}>
        <div className="judging">
          <div className="spinner" />
          <p>{judging.log?.slice(-1)[0] || 'sending transcript…'}</p>
          <p className="dim small">{judging.elapsed}s so far.</p>
        </div>
      </Modal>
    )
  }
  if (!v) return null

  const cls = v.correct ? (v.evidence_backed ? 'ok' : 'warn') : 'bad'
  const headline = v.correct
    ? (v.evidence_backed ? 'CASE CLOSED' : 'RIGHT NAME, THIN CASE')
    : 'WRONG'
  const H = (x) => humanDetail(x, state)

  return (
    <Modal title="Verdict" onClose={onClose} wide>
      <div className={`verdict-head ${cls}`}>
        <div className="verdict-row">
          <div className="verdict-big">{headline}</div>
          <div className={`grade grade-${v.grade}`}>{v.grade}</div>
        </div>
        <div>You accused <b>{v.accused_name}</b>. The culprit was <b>{v.culprit_name}</b>.</div>
        <div className="dim">{v.points > 0 ? '+' : ''}{v.points} pts · final score <b>{v.final_score}</b></div>
      </div>

      <p className="brief">{H(v.verdict)}</p>

      <div className="tabs">
        {['debrief', 'evidence', 'solution'].map(t => (
          <button key={t} className={tab === t ? 'picked' : ''} onClick={() => setTab(t)}>{t}</button>
        ))}
      </div>

      {tab === 'debrief' && (
        <>
          {v.what_went_wrong?.length > 0 && (
            <>
              <h4>{v.correct ? 'What was missing' : 'What went wrong'}</h4>
              <ul className="plain">{v.what_went_wrong.map((x, i) => <li key={i}>{H(x)}</li>)}</ul>
            </>
          )}
          {v.missed_opportunities?.length > 0 && (
            <>
              <h4>Catchable, and you moved on</h4>
              <ul className="plain">{v.missed_opportunities.map((x, i) => <li key={i}>{H(x)}</li>)}</ul>
            </>
          )}
          {v.how_to_improve?.length > 0 && (
            <>
              <h4>Do this next time</h4>
              <ol className="plain advice">{v.how_to_improve.map((x, i) => <li key={i}>{H(x)}</li>)}</ol>
            </>
          )}
          {v.best_moment && <p><b>Best moment.</b> {H(v.best_moment)}</p>}
        </>
      )}

      {tab === 'evidence' && (
        <div className="grid2">
          <div>
            <h4>Surfaced</h4>
            {v.contradictions_surfaced?.length
              ? <ul className="plain">{v.contradictions_surfaced.map((x, i) => <li key={i}>{H(x)}</li>)}</ul>
              : <p className="dim small">You did not get a single lie out.</p>}
          </div>
          <div>
            <h4>Missed</h4>
            {v.contradictions_missed?.length
              ? <ul className="plain">{v.contradictions_missed.map((x, i) => <li key={i}>{H(x)}</li>)}</ul>
              : <p className="dim small">You got them all.</p>}
          </div>
          {v.wasted_lines?.length > 0 && (
            <div style={{ gridColumn: '1 / -1' }}>
              <h4>Wasted</h4>
              <ul className="plain">{v.wasted_lines.map((x, i) => <li key={i} className="dim">{H(x)}</li>)}</ul>
            </div>
          )}
        </div>
      )}

      {tab === 'solution' && (
        <>
          <h4>How the case actually breaks</h4>
          <p className="lab-quote">{H(v.the_solution) || 'not available'}</p>
        </>
      )}

      <div className="dim small" style={{ marginTop: 10 }}>
        judged by {v.judged_by}
        {v.judged_by === 'offline' && ' — set a DeepSeek key for a proper read of your questioning'}
      </div>
      <div className="row" style={{ marginTop: 12 }}>
        <button className="primary" onClick={onNewCase}>new case</button>
        <button onClick={onClose}>close</button>
      </div>
    </Modal>
  )
}

/* ================================================================== */
/*  New case                                                            */
/* ================================================================== */
export function NewCase ({ onClose, onLoaded, busy, setBusy, setErr }) {
  const [lib, setLib] = useState(null)
  const [caseId, setCaseId] = useState('')
  const [diff, setDiff] = useState('normal')
  const [n, setN] = useState(3)
  const [job, setJob] = useState(null)

  useEffect(() => {
    api('cases').then(d => { setLib(d); setCaseId(d.current) }).catch(e => setErr(String(e)))
  }, [])

  const load = async () => {
    setBusy(true); setErr(null)
    try { await api('case/load', { case_id: caseId, difficulty: diff }); onLoaded() }
    catch (e) { setErr(String(e.message || e)) }
    finally { setBusy(false) }
  }

  const generate = async () => {
    setBusy(true); setErr(null)
    try {
      const { job: id } = await api('case/generate', { n_suspects: n, difficulty: diff })
      const j = await waitJob(id, setJob)
      if (j.status === 'failed') throw new Error(j.error)
      if (j.result.generated_by === 'offline') {
        setErr('That case was built locally, not by DeepSeek.')
      }
      await api('case/load', { case_id: j.result.case_id, difficulty: diff })
      onLoaded()
    } catch (e) { setErr(String(e.message || e)) }
    finally { setBusy(false); setJob(null) }
  }

  if (!lib) return <Modal title="New case" onClose={onClose}><p className="dim">loading library…</p></Modal>

  return (
    <Modal title="New case" onClose={onClose}>
      <h4>Difficulty</h4>
      <div className="row wrap">
        {Object.entries(lib.difficulties).map(([k, label]) => (
          <button key={k} className={diff === k ? 'picked' : ''} onClick={() => setDiff(k)}>{label}</button>
        ))}
      </div>
      <p className="dim small">
        Easy 15 turns, free rewinds. Normal 10 turns, rewind keeps the turn.
        Hard 7 turns, question types hidden, rewind costs two.
      </p>

      <h4>From the library</h4>
      <div className="row">
        <select value={caseId} onChange={e => setCaseId(e.target.value)}>
          {lib.cases.map(c => (
            <option key={c.case_id} value={c.case_id}>
              {c.title} — {c.n_suspects} suspects ({c.generated_by})
            </option>
          ))}
        </select>
        <button className="primary" disabled={busy} onClick={load}>play</button>
      </div>

      <h4>Or write a new one</h4>
      <div className="row">
        <label className="dim small">suspects</label>
        <input type="number" min="2" max="8" value={n} style={{ width: 60, flex: 'none' }}
               onChange={e => setN(Number(e.target.value))} />
        <button className="primary" disabled={busy} onClick={generate}>
          generate with DeepSeek
        </button>
      </div>
      {job && (
        <div className="judging">
          <div className="spinner" />
          <p>{job.log?.slice(-1)[0] || 'starting…'}</p>
          <p className="dim small">{job.elapsed}s · the validator may send it back for repairs</p>
        </div>
      )}
      <p className="dim small">
        deepseek-reasoner writes it, the validator checks it holds together,
        and rejects it until it does. One to three minutes. Without an API key
        you get a deterministic offline case instead, and it says so.
      </p>
    </Modal>
  )
}

/* ================================================================== */
/*  Help                                                                */
/* ================================================================== */
export function Help ({ onClose }) {
  const [text, setText] = useState('')
  useEffect(() => {
    fetch('/api/manual').then(r => r.text()).then(setText).catch(() => setText('manual unavailable'))
  }, [])
  return (
    <Modal title="How to play" onClose={onClose} wide>
      <pre className="manual">{text}</pre>
    </Modal>
  )
}

/* ================================================================== */
/*  The machine - branch tree and cache panel                           */
/* ================================================================== */
const TIER_COLOR = { gpu: 'var(--ok)', int8: 'var(--warn)', cpu: 'var(--dim)', dropped: 'var(--bad)' }

export function BranchTree ({ suspect, candidates, chosen }) {
  const turns = suspect ? suspect.turns : []
  const rowH = 28, tx = 24
  const h = Math.max(110, turns.length * rowH + (candidates.length ? 120 : 40))
  return (
    <svg viewBox={`0 0 250 ${h}`} className="tree">
      {turns.map((t, i) => {
        const y = 16 + i * rowH
        return (
          <g key={i}>
            {i > 0 && <line x1={tx} y1={y - rowH} x2={tx} y2={y} stroke="var(--line)" strokeWidth="1.5" />}
            <circle cx={tx} cy={y} r="5" fill={TIER_COLOR[t.snapshot_tier] || 'var(--dim)'} />
            <text x={tx + 12} y={y + 4} className="lbl">t{t.index} · {t.snapshot_tier}{t.band ? ` · ${t.band}` : ''}</text>
          </g>
        )
      })}
      {candidates.length > 0 && (() => {
        const baseY = 16 + Math.max(0, turns.length - 1) * rowH
        const fy = baseY + rowH
        return (
          <g>
            <line x1={tx} y1={baseY} x2={tx} y2={fy} stroke="var(--line)" strokeWidth="1.5" />
            <circle cx={tx} cy={fy} r="6" fill="var(--accent)" />
            <text x={tx + 12} y={fy + 4} className="lbl">fork ×{candidates.length}</text>
            {candidates.map((c, i) => {
              const y = fy + 24 + i * 24
              const picked = chosen === i
              return (
                <g key={i}>
                  <line x1={tx} y1={fy} x2={80} y2={y} stroke={picked ? 'var(--ok)' : 'var(--line)'} strokeWidth={picked ? 2 : 1.2} />
                  <circle cx="80" cy={y} r="4.5" fill={picked ? 'var(--ok)' : 'var(--dim)'} />
                  <text x="90" y={y + 4} className="lbl">{c.kind.replace('_', ' ')}</text>
                </g>
              )
            })}
          </g>
        )
      })()}
    </svg>
  )
}

export function Machine ({ state, suspect, candidates, chosen }) {
  const [open, setOpen] = useState(true)
  const a = state.accounting, st = state.store
  const pct = st.budget_bytes ? Math.min(100, Math.round(100 * st.gpu_bytes / st.budget_bytes)) : 0
  return (
    <aside className={`machine ${open ? '' : 'closed'}`}>
      <button className="machine-toggle" onClick={() => setOpen(o => !o)}>
        {open ? '▸ hide the machine' : '◂ the machine'}
      </button>
      {open && <>
        <div className="panel">
          <div className="panel-title">branch tree — {suspect.name}</div>
          <BranchTree suspect={suspect} candidates={candidates} chosen={chosen} />
        </div>
        <div className="grid2">
          <div className="stat"><div className="stat-label">prefill tokens</div><div className="stat-value">{a.prefill_tokens.toLocaleString()}</div></div>
          <div className="stat"><div className="stat-label">saved on shared</div><div className="stat-value">{a.tokens_saved_on_shared_block.toLocaleString()}</div></div>
          <div className="stat"><div className="stat-label">forks / crops</div><div className="stat-value">{a.forks} / {a.crops}</div></div>
          <div className="stat"><div className="stat-label">evictions</div><div className="stat-value">{a.evictions}</div></div>
        </div>
        <div className="panel">
          <div className="panel-title">snapshot store</div>
          <div className="bar tall"><div className="fill ok" style={{ width: `${pct}%` }} /></div>
          <div className="mono small">{st.report}</div>
          <div className="mono small dim">hits gpu {st.stats.gpu} · int8 {st.stats.int8} · cpu {st.stats.cpu} · miss {st.stats.misses}</div>
        </div>
        <div className="panel">
          <div className="panel-title">cache log</div>
          <div className="log">
            {state.log.slice().reverse().map((e, i) => (
              <div key={i} className="mono small">
                <span className={`kind ${e.kind}`}>{e.kind}</span>
                <span className="dim"> {String(e.tokens).padStart(5)} </span>{e.label}
              </div>
            ))}
          </div>
        </div>
      </>}
    </aside>
  )
}