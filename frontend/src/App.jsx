import React, { useCallback, useEffect, useState } from 'react'
import { api, waitJob, last } from './api.js'
import { CaseFile, Dossier, Verdict, NewCase, Help, Machine } from './Panels.jsx'

/* ------------------------------------------------------------------ */
function SuspectCard ({ s, active, onSelect, onDossier }) {
  const pct = Math.round(s.composure * 100)
  const tone = s.composure > 0.6 ? 'ok' : s.composure > 0.35 ? 'warn' : 'bad'
  return (
    <div className={`card ${active ? 'active' : ''}`}>
      <button className="card-main" onClick={onSelect}>
        <div className="card-name">{s.name}</div>
        <div className="card-role">{s.role}</div>
        <div className="bar"><div className={`fill ${tone}`} style={{ width: `${pct}%` }} /></div>
        <div className="card-meta">
          {s.turns.length} turns
          {s.hits.length > 0 && <span className="flag"> · {s.hits.length} caught</span>}
        </div>
      </button>
      <button className="mini dossier-btn" onClick={onDossier}>dossier</button>
    </div>
  )
}

/* ------------------------------------------------------------------ */
function Hud ({ game, file, onCase, onHelp, onNew }) {
  const left = game.turns_left
  const tone = left > game.rules.turns * 0.5 ? 'ok' : left > 2 ? 'warn' : 'bad'
  return (
    <header className="hud">
      <div className="hud-left">
        <div className="title">{file.title}</div>
        <div className="dim small">{game.rules.label} · deepseek {game.deepseek}</div>
      </div>
      <div className="hud-mid">
        <div className="stat-inline">
          <span className="stat-label">turns</span>
          <span className={`stat-value ${tone}`}>{left}</span>
          <span className="dim">/ {game.rules.turns}</span>
        </div>
        <div className="stat-inline">
          <span className="stat-label">score</span>
          <span className="stat-value">{game.score}</span>
        </div>
        <div className="stat-inline">
          <span className="stat-label">caught</span>
          <span className="stat-value">{game.contradictions_found}</span>
          <span className="dim">/ {game.contradictions_available}</span>
        </div>
      </div>
      <div className="hud-right">
        <button onClick={onCase}>📁 case file</button>
        <button onClick={onNew}>new case</button>
        <button onClick={onHelp}>?</button>
      </div>
    </header>
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
  const [reveal, setReveal] = useState(null)         // last committed answer
  const [tools, setTools] = useState([])
  const [toolOut, setToolOut] = useState(null)
  const [judging, setJudging] = useState(null)
  const [modal, setModal] = useState(null)           // 'case' | 'dossier' | 'verdict' | 'new' | 'help'
  const [dossierId, setDossierId] = useState(null)

  const refresh = useCallback(async () => {
    const s = await api('state')
    setState(s)
    setSel(cur => cur && s.suspects.some(x => x.id === cur) ? cur : s.suspects[0]?.id)
    return s
  }, [])

  useEffect(() => { refresh().catch(e => setErr(String(e))) }, [refresh])
  useEffect(() => {
    api('tools').then(d => setTools(d.tools)).catch(() => setTools([]))
  }, [state?.case?.case_id])

  const run = async (fn) => {
    setBusy(true); setErr(null)
    try { await fn(); await refresh() }
    catch (e) { setErr(String(e.message || e)) }
    finally { setBusy(false) }
  }

  if (!state) return <div className="boot">{err ? `error: ${err}` : 'loading the model…'}</div>

  const { game, case: file } = state
  const suspect = state.suspects.find(s => s.id === sel) || state.suspects[0]
  const candidates = state.candidates[suspect.id] || []
  const hints = game.rules.hints
  const closed = game.finished || game.turns_left <= 0

  /* ------------------------------------------------------- actions */
  const explore = () => run(() => api('explore', { suspect_id: suspect.id }))

  const commit = () => run(async () => {
    const r = await api('commit', { suspect_id: suspect.id, index: chosen })
    setReveal({ ...r, name: suspect.name })
    setChosen(null)
  })

  const askOwn = () => run(async () => {
    const r = await api('ask', { suspect_id: suspect.id, question: typed })
    setReveal({ ...r, name: suspect.name, kind: 'your question', band: null })
    setTyped('')
  })

  const quote = (src) => run(async () => {
    const r = await api('quote', {
      from_suspect: src.id, turn_index: src.turns.length - 1, to_suspect: suspect.id })
    setReveal({ ...r, name: suspect.name, kind: `quoted ${last(src.name)}`, band: null })
  })

  const rewind = (turnIndex) => run(async () => {
    await api('rewind', { suspect_id: dossierId || suspect.id, turn_index: turnIndex })
    setReveal(null); setModal(null)
  })

  const tool = (t) => run(async () => {
    const r = await api('tool', { name: t.name, args: argsFor(t.name, suspect, state) })
    setToolOut({ name: t.name.replace('_tool', ''), ...r })
  })

  const accuse = (s) => run(async () => {
    const r = await api('accuse', { suspect_id: s.id })
    setModal('verdict')
    setJudging({ elapsed: 0 })
    const j = await waitJob(r.job, (jj) => setJudging(jj))
    setJudging(null)
    if (j.status === 'failed') throw new Error(j.error)
  })

  /* ------------------------------------------------------- render */
  return (
    <div className="app">
      <Hud game={game} file={file}
           onCase={() => setModal('case')}
           onHelp={() => setModal('help')}
           onNew={() => setModal('new')} />

      <div className="cols">
        <section className="room">
          <div className="cards">
            {state.suspects.map(s => (
              <SuspectCard key={s.id} s={s} active={s.id === suspect.id}
                           onSelect={() => { setSel(s.id); setChosen(null); setReveal(null) }}
                           onDossier={() => { setDossierId(s.id); setModal('dossier') }} />
            ))}
          </div>

          {/* ---------------- the last exchange, revealed ---------------- */}
          <div className="exchange">
            {reveal ? (
              <>
                <div className="q">
                  {hints && reveal.kind && <span className="chip">{reveal.kind.replace('_', ' ')}</span>}
                  {reveal.question || suspect.turns.slice(-1)[0]?.question}
                </div>
                <div className="a">{reveal.name}: {reveal.answer}</div>
                <div className="row wrap outcome">
                  {reveal.band && <span className={`band ${reveal.band}`}>{reveal.band} · {reveal.score.toFixed(2)}</span>}
                  {reveal.new_facts > 0 && <span className="tag ok">+{reveal.new_facts} to the board</span>}
                  {reveal.contradiction && <span className="tag bad">caught: {reveal.contradiction}</span>}
                  {reveal.granted?.length > 0 && <span className="tag warn">now knows {reveal.granted.length} more</span>}
                  {reveal.new_facts === 0 && !reveal.contradiction && !reveal.granted?.length &&
                    <span className="tag dim">nothing new</span>}
                </div>
              </>
            ) : suspect.turns.length ? (
              <div className="dim small">
                {suspect.turns.length} exchange{suspect.turns.length > 1 ? 's' : ''} so far — open the dossier to read them.
              </div>
            ) : (
              <div className="dim small">{suspect.name} has not been questioned yet.</div>
            )}
          </div>

          {/* ---------------- choose a question ---------------- */}
          {closed ? (
            <div className="closed">
              {game.finished
                ? 'The case is closed.'
                : 'Out of turns. You must accuse someone.'}
            </div>
          ) : candidates.length > 0 ? (
            <div className="options">
              <div className="panel-title">three lines of questioning — pick one</div>
              {candidates.map(c => (
                <button key={c.index} disabled={busy}
                        className={`option ${chosen === c.index ? 'picked' : ''}`}
                        onClick={() => setChosen(c.index)}>
                  {hints && <span className="chip">{c.kind.replace('_', ' ')}</span>}
                  <span className="q">{c.question}</span>
                </button>
              ))}
              <button className="primary" disabled={busy || chosen === null} onClick={commit}>
                ask it
              </button>
            </div>
          ) : (
            <button className="primary big" disabled={busy} onClick={explore}>
              {busy ? 'thinking…' : `question ${last(suspect.name)}`}
            </button>
          )}

          <div className="row">
            <input value={typed} placeholder="or ask in your own words" disabled={closed}
                   onChange={e => setTyped(e.target.value)}
                   onKeyDown={e => e.key === 'Enter' && typed.trim() && askOwn()} />
            <button disabled={busy || closed || !typed.trim()} onClick={askOwn}>ask</button>
          </div>

          <div className="row wrap">
            {state.suspects.filter(s => s.id !== suspect.id && s.turns.length > 0).map(src => (
              <button key={src.id} disabled={busy || closed} onClick={() => quote(src)}>
                quote {last(src.name)} at {last(suspect.name)}
              </button>
            ))}
          </div>

          {tools.length > 0 && (
            <div className="lab">
              <div className="panel-title">the lab — costs a turn</div>
              <div className="row wrap">
                {tools.map(t => (
                  <button key={t.name} disabled={busy || closed} onClick={() => tool(t)}>
                    {t.name.replace('_tool', '').replace(/_/g, ' ')}
                  </button>
                ))}
              </div>
              {toolOut && (
                <div className="toolout">
                  <div className="row wrap">
                    <b>{toolOut.name}</b>
                    {toolOut.flag && <span className="tag bad">{toolOut.flag}</span>}
                    {toolOut.new_facts > 0 && <span className="tag ok">+{toolOut.new_facts} to the board</span>}
                  </div>
                  <pre className="mono small">{JSON.stringify(toolOut.result, null, 2)}</pre>
                </div>
              )}
            </div>
          )}

          <div className="row wrap accuse-row">
            <span className="dim small">accuse:</span>
            {state.suspects.map(s => (
              <button key={s.id} className="danger" disabled={busy || game.finished}
                      onClick={() => accuse(s)}>{last(s.name)}</button>
            ))}
          </div>

          {err && <div className="note bad">{err}</div>}
        </section>

        <Machine state={state} suspect={suspect} candidates={candidates} chosen={chosen} />
      </div>

      {modal === 'case' && <CaseFile file={file} onClose={() => setModal(null)} />}
      {modal === 'dossier' && dossierId && (
        <Dossier s={state.suspects.find(s => s.id === dossierId)} busy={busy} hints={hints}
                 onClose={() => setModal(null)} onRewind={rewind} />
      )}
      {modal === 'verdict' && (
        <Verdict v={game.verdict} judging={judging}
                 onNewCase={() => setModal('new')} onClose={() => setModal(null)} />
      )}
      {modal === 'new' && (
        <NewCase busy={busy} setBusy={setBusy} setErr={setErr}
                 onClose={() => setModal(null)}
                 onLoaded={async () => { setModal(null); setReveal(null); setChosen(null); setToolOut(null); await refresh() }} />
      )}
      {modal === 'help' && <Help onClose={() => setModal(null)} />}
    </div>
  )
}

/* Each forensic tool declares different parameters and the server
   validates them, so every one gets its own branch. The scene comes from
   the backend rather than being hard-coded. */
function argsFor (toolName, suspect, state) {
  const name = toolName.replace(/_tool$/, '')
  const scene = state.case.scene || ''
  const at = state.case.window?.from || '21:00'
  const lastAnswer = suspect.turns.length ? suspect.turns[suspect.turns.length - 1].answer : ''
  switch (name) {
    case 'check_alibi':      return { suspect_id: suspect.id, place: scene, at }
    case 'who_was_at':       return { place: scene, at }
    case 'movements_of':     return { suspect_id: suspect.id }
    case 'verify_statement': return { suspect_id: suspect.id, statement: lastAnswer }
    case 'lookup_evidence':  return { query: scene }
    default:                 return { suspect_id: suspect.id }
  }
}