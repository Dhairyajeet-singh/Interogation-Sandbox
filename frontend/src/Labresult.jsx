import React, { useState } from 'react'

/* ==================================================================
   Turning forensic results into something a detective would read.

   The lab used to dump raw JSON on screen, which is fine for debugging
   and useless in a game. Each tool returns a different shape, so each
   gets its own renderer, with a fallback for anything unrecognised.
   ================================================================== */

/* entity ids are machine-readable; people are not */
export function pretty (id) {
  if (typeof id !== 'string') return String(id ?? '')
  return id
    .replace(/^per_/, '')
    .replace(/^loc_/, '')
    .replace(/^ev_/, '')
    .replace(/^time_/, '')
    .replace(/_/g, ' ')
}

export function nameOf (id, state) {
  const s = state?.case?.suspects?.find(x => `per_${x.id}` === id || x.id === id)
  if (s) return s.name
  return pretty(id)
}

function Verdictly ({ verdict }) {
  const cls = verdict === 'supported' ? 'ok'
    : verdict === 'contradicted' ? 'bad' : 'dim'
  const word = verdict === 'supported' ? 'CHECKS OUT'
    : verdict === 'contradicted' ? 'DOES NOT CHECK OUT' : 'NO RECORD'
  return <span className={`tag ${cls}`}>{word}</span>
}

function Row ({ label, children }) {
  return (
    <div className="lab-row">
      <span className="lab-label">{label}</span>
      <span>{children}</span>
    </div>
  )
}

/* ------------------------------------------------------------------ */
function CheckAlibi ({ r, state }) {
  return (
    <>
      <div className="lab-headline"><Verdictly verdict={r.verdict} /></div>
      <Row label="subject">{nameOf(r.suspect, state)}</Row>
      <Row label="claimed at">{pretty(r.place)}, {r.at}</Row>
      <Row label="the record says">{humanDetail(r.detail, state)}</Row>
    </>
  )
}

function WhoWasAt ({ r, state }) {
  if (!r.people?.length) {
    return <p className="lab-empty">Nobody is on record at {pretty(r.place)}
      {r.at ? ` at ${r.at}` : ''}.</p>
  }
  return (
    <>
      <div className="lab-headline">
        {r.people.length} on record at <b>{pretty(r.place)}</b>
        {r.at ? <> at <b>{r.at}</b></> : null}
      </div>
      <table className="lab-table">
        <tbody>
          {r.people.map((p, i) => (
            <tr key={i}>
              <td><b>{nameOf(p.suspect, state)}</b></td>
              <td className="mono">{p.from}–{p.to}</td>
              <td className="dim small">{p.verified_by
                ? `verified by ${pretty(p.verified_by)}` : 'unverified'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  )
}

function Movements ({ r, state }) {
  if (!r.movements?.length) {
    return <p className="lab-empty">No recorded movements for {nameOf(r.suspect, state)}.</p>
  }
  return (
    <>
      <div className="lab-headline">Movements — <b>{nameOf(r.suspect, state)}</b></div>
      <ol className="lab-timeline">
        {r.movements.map((m, i) => (
          <li key={i}>
            <span className="mono time">{m.from}–{m.to}</span>
            <span className="place">{pretty(m.place)}</span>
            {m.verified_by && <span className="tag ok small">verified</span>}
          </li>
        ))}
      </ol>
    </>
  )
}

function Evidence ({ r }) {
  if (r.error) {
    return (
      <>
        <p className="lab-empty">{r.error}</p>
        {r.known_ids && <p className="dim small">
          on file: {r.known_ids.map(pretty).join(', ')}</p>}
      </>
    )
  }
  return (
    <>
      <div className="lab-headline">
        <span className="chip">{r.type}</span> {pretty(r.id)}
      </div>
      <p className="lab-quote">{r.text}</p>
      {r.aliases?.length > 0 && (
        <p className="dim small">also called: {r.aliases.join(' · ')}</p>
      )}
    </>
  )
}

function VerifyStatement ({ r, state }) {
  const checks = r.checks || []
  return (
    <>
      <div className="lab-headline">
        {r.contradicted
          ? <span className="tag bad">STATEMENT DOES NOT HOLD</span>
          : checks.length
            ? <span className="tag ok">NOTHING CONTRADICTED</span>
            : <span className="tag dim">NOTHING CHECKABLE SAID</span>}
      </div>
      {checks.length === 0 && (
        <p className="lab-empty">
          They gave no claim the record can test — no place and time together.
        </p>
      )}
      {checks.map((c, i) => (
        <div key={i} className="lab-check">
          <Verdictly verdict={c.verdict} />
          <span> claimed <b>{pretty(c.place)}</b> at <b>{c.at}</b></span>
          <div className="dim small">{humanDetail(c.detail, state)}</div>
        </div>
      ))}
      {r.entities_mentioned?.length > 0 && (
        <p className="dim small">
          mentioned: {r.entities_mentioned.map(pretty).join(' · ')}
        </p>
      )}
    </>
  )
}

/* the server phrases details with raw ids; swap them for names.
   Exported because contradiction notes need it too - "said loc_study
   earlier, now says loc_observatory" is not a sentence. */
export function humanDetail (detail, state) {
  if (!detail) return ''
  return String(detail).replace(/\b(per|loc|ev|time)_[a-z0-9_]+/g,
    (m) => m.startsWith('per_') ? nameOf(m, state) : pretty(m))
}

const RENDERERS = {
  check_alibi: CheckAlibi,
  who_was_at: WhoWasAt,
  movements_of: Movements,
  lookup_evidence: Evidence,
  verify_statement: VerifyStatement,
}

/* ------------------------------------------------------------------ */
export default function LabResult ({ out, state }) {
  const [raw, setRaw] = useState(false)
  if (!out) return null

  const Renderer = RENDERERS[out.name]
  const title = out.name.replace(/_/g, ' ')

  return (
    <div className="lab-result">
      <div className="lab-head">
        <span className="lab-title">{title}</span>
        {out.flag && <span className="tag bad">{humanDetail(out.flag, state)}</span>}
        {out.new_facts > 0 && <span className="tag ok">+{out.new_facts} to the board</span>}
        <button className="mini" onClick={() => setRaw(r => !r)}>
          {raw ? 'report' : 'raw'}
        </button>
      </div>

      {raw || !Renderer
        ? <pre className="mono small lab-raw">{JSON.stringify(out.result, null, 2)}</pre>
        : <div className="lab-body"><Renderer r={out.result} state={state} /></div>}
    </div>
  )
}