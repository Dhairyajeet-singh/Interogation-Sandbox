# Branching Interrogation Sandbox — stages 0 to 9

A text detective game where the AI suspects share one prefilled KV cache,
can be rewound by cropping that cache, and can be cheaply forked so the
game can test several questions before committing to one.

The game is the demo. The subject is direct manipulation of the
transformer's key/value attention state.

## Install

```
pip install transformers accelerate torch
pip install rank_bm25 networkx sentence-transformers
pip install langgraph mcp fastapi "uvicorn[standard]" httpx
cd frontend && npm install && npm run build && cd ..
```

No transformers pin needed. `cache_ops.normalise()` / `denormalise()`
handle all three cache formats the library has used, including the 5.x
`DynamicCache` that has no legacy methods and is not subscriptable.

## Files

| file | stage | what it is |
|---|---|---|
| `cache_ops.py` | 0 | fork, crop, snapshot, restore, prefill, decode, cross-version cache compat |
| `Model_setup.py` | — | loads the model; defines comparison tolerances |
| `Case_Ashfield.json` | 3 | the case: entities, aliases, public/private facts, timeline, knows-lists, planted lies |
| `Extractor.py` | 3b | reads an answer, returns which case facts it mentions. Pure string matching, no model |
| `Retrieval.py` | 3 | BM25 + embeddings + timeline graph, fused with RRF, filtered by the knows-list |
| `Suspects.py` | 2,3 | the Suspect class: LCP cache reuse, per-turn snapshots, crop-based rewind |
| `Branching.py` | 5 | three-option questioning with mechanical scoring |
| `Snapshots.py` | 6 | three-tier snapshot store with an LRU eviction policy |
| `Interrogation.py` | 7 | composure, cross-suspect quoting, contradiction log |
| `Mcp_server.py` | 8 | forensic tools over MCP — stdio and HTTP |
| `Mcp_client.py` | 8 | discovers and calls those tools |
| `Api.py` | 9 | FastAPI backend holding the model and every cache |
| `Game.py` | 4 | the LangGraph interrogation loop (terminal) |
| `frontend/` | 9 | Vite + React UI |

## Running

```
py Mcp_server.py                     # stdio, what an MCP client spawns
py Mcp_server.py --http --port 8765  # HTTP, easy to curl
py Api.py                            # backend + built frontend on :8000
py Game.py                           # terminal version
```

For frontend development, run `py Api.py` and `npm run dev` in `frontend/`
side by side; Vite proxies `/api` to port 8000.

### One process, one worker, no reload

The KV caches live in the backend's memory.

```
py Api.py                       correct
uvicorn Api:app --reload        WRONG — wipes the model on every save
uvicorn Api:app --workers 4     WRONG — each worker gets its own caches
```

This is also why the project cannot deploy to anything that scales to zero.

## Tests

```
py Mockrun.py            logic only, no GPU, no model download
py Test_Extractor.py     stage 3b — 20 labelled answers
py Test_Retrieval.py     stage 3  — search + knowledge filter
py test_cache_ops.py     stage 0  — cache primitives
py Test_Rewind.py        stage 2  — rewind correctness
py Test_Branching.py     stage 5  — fork cost + score stability
py Test_Stages678.py     stages 6,7,8
py Test_Api.py           stage 9  — every endpoint, live model
```

All eight verified against real Qwen2.5-0.5B-Instruct on transformers 5.17.

## How cache reuse works

Not by slicing the prompt at fixed indices — that broke, because BPE
merges the last character of one block with the first of the next.

Instead the code compares token id sequences and reuses however many
leading tokens genuinely match. Longest-common-prefix reuse, the same
mechanism vLLM and SGLang use internally.

```
cached:  [a b c d e f g]      what the cache already holds
wanted:  [a b c d X Y]        what this turn needs
         ^^^^^^^              4 tokens in common
-> crop the cache to 4, prefill [X Y]
```

`compute_shared_len()` finds the longest prefix all suspects agree on;
those tokens are prefilled once and forked. Each turn then extends by
appending token ids directly rather than re-rendering the template,
because re-tokenising the model's own answers does not round-trip.

## Stage 6 — three tiers

Ordered by how expensive it is to get a snapshot back:

| tier | where | lossless | cost to restore |
|---|---|---|---|
| `gpu` | VRAM, full precision | yes | none |
| `int8` | VRAM, quantised | no (~0.4% of value range) | a dequantise |
| `cpu` | host RAM, full precision | yes, if it came straight from gpu | a PCIe transfer |
| `dropped` | gone | — | recompute from scratch |

Demotion is least-recently-used. A snapshot demoted *through* int8 keeps
that loss when it lands on CPU — you cannot recover precision already
thrown away. The tests check both paths separately.

`SNAPSHOT_BUDGET_MB` in `Api.py` is deliberately small so eviction
happens during a demo and the UI has something to show.

## Stage 7 — the sharp test

Telling suspect A what B said grants A the entity knowledge behind the
quote, which changes what retrieval may hand them. The grant is recorded
on the turn, so rewinding past it takes the knowledge away again.

A prompt-level "forget that" could never do this. It is the strongest
evidence in the project that the rewind is real state restoration.

Composure feeds sampling temperature: calm suspects decode greedily,
cornered ones sample. Speculative previews always use temperature 0
regardless, because stage 5's scores must be reproducible.

## Scoring

No model judges anything. Three counted quantities:

- **coverage** — entity ids in the answer not seen before in this transcript
- **verifiable** — claims the timeline could adjudicate (time / place / person)
- **contradiction** — 1 if it conflicts with an earlier statement or the timeline

Weights live in `Branching.WEIGHTS`, to be grid-searched against ground
truth in stage 12. The UI shows a band, not the raw float — `0.71` claims
a precision the score does not have.

## Exact vs tolerant comparisons

**Exact (`== 0.0`)** — fork, crop, snapshot roundtrip. These copy or slice
numbers that already exist.

**Tolerant** — anything where the model recomputes. Prefilling 7 tokens
onto 14 cached is a different sequence of float reductions than
prefilling 21 at once. Measured on real Qwen2.5-0.5B: 8.7e-05 in fp32. A
real bug would be order-of-magnitude, not fifth-decimal.

Scores are reproducible **on a given machine**, not across hardware —
fp16 and fp32 generate slightly different text, so the extractor counts
different entities.

## Still to do

- Stage 10 — case generator and validator (external API writes cases into this schema)
- Stage 11 — the judge (transcript + ground truth → verdict)
- Stage 12 — experiments: tokens saved vs suspect count, branching vs single-shot, eviction hit rate, weight ablation

---

# Stages 10, 11, 12

## DeepSeek setup

Both the generator and the judge use `deepseek-reasoner` through the
OpenAI-compatible endpoint. The key comes from the environment — never
hard-code it, never commit it.

```
setx DEEPSEEK_API_KEY sk-...        Windows, then reopen the shell
export DEEPSEEK_API_KEY=sk-...      Linux / macOS
pip install openai matplotlib
```

With no key set, every call falls back to a deterministic offline path
and the artefact is stamped `generated_by="offline"` so it can never be
mistaken for model output. That is deliberate: a dead API or an exhausted
quota must not take down a demo, and every test has to run without a
network.

## Stage 10 — `Generator.py`

```
py Generator.py --n 10 --suspects 3 4 5     write cases to cases/
py Generator.py --offline --n 3             no API, deterministic
```

The validator is the substance. Asking a model for "a murder mystery"
gets prose that collapses the moment a detective pushes on it, so nothing
is accepted until it holds together:

| check | what it catches |
|---|---|
| structural | missing fields, malformed entity ids |
| referential | ids pointing at entities that do not exist |
| temporal | bad clocks, backwards ranges, one person in two places at once |
| opportunity | a culprit who is never at the scene during the window |
| solvable | a lie only exposable by facts the liar already knows |
| extractable | fewer than 3 aliases, or an alias claimed by two entities |

On failure the errors are handed back and a repair is requested, up to
three rounds. Repair is cheaper than regeneration because the model keeps
what was already fine.

**The validator rejected my own hand-written case on first run** — two
entities with too few aliases, and Rourke's lie only exposable by facts he
already knew, which made confronting him with it pointless. `ev_boot_prints`
was added to fix it. That is the validator doing its job.

## Stage 11 — `Judge.py`

Sends the transcript, the ground truth, and the accusation to the model
and gets back a structured verdict.

The distinction worth having is **correct** versus **evidence-backed**. A
right name with no supporting work is a lucky guess, and it looks
identical to earned deduction in the final answer while looking nothing
like it in the transcript. Separating them needs a reader. Solve rate and
evidence-backed solve rate are reported separately.

The judge sees the truth; the suspects never do. That asymmetry is why
the true case lives in its own store the game never reads from — and
there is a test asserting `SECRET:` text never reaches the transcript.

## Stage 12 — `Experiments.py`

```
py Experiments.py --scale small      ~10 min, check it runs
py Experiments.py --scale large      the real thing, a few hours
py Experiments.py --only e1 e3       just some
py Experiments.py --plots            redraw from saved results
```

Checkpointed after every unit of work, so a long run resumes where it
stopped.

| | question | output |
|---|---|---|
| E1 | how much does the cache save? | prefill tokens, forked vs naive, by suspect count |
| E2 | does trying three questions beat asking one? | contradiction recall for argmax / random / single |
| E3 | what does the memory budget cost? | hit rate and tier mix as the GPU budget shrinks |
| E4 | where did those weights come from? | grid search against ground truth |

### Measured so far, locally

E1, and this one is solid:

| suspects | forked | naive | saved |
|---|---|---|---|
| 2 | 275 | 411 | 33% |
| 3 | 335 | 607 | 45% |
| 4 | 398 | 806 | 51% |

E3, the eviction ladder under pressure:

```
512MB   gpu 3  int8 0  cpu 0
 16MB   gpu 2  int8 0  cpu 1
  8MB   gpu 0  int8 1  cpu 2
  4MB   gpu 0  int8 0  cpu 3
```

### The caveat that matters for E2 and E4

On the formulaic offline-generated cases, **every policy scored zero
contradictions**, which makes E2 and E4 answer nothing. The cause is case
construction, not the model: those lies are weakly built.

On the hand-written case, the same experiment discriminates clearly:

```
argmax  found 2/3
single  found 0/3
```

So `load_cases()` now puts `Case_Ashfield.json` first, and the report
writes an explicit warning into E2 and E4 if recall comes out uniformly
zero rather than letting an empty table look like a result.

Two honest notes for the write-up:

- At 0.5B the detector needs a suspect to name a specific place *and* a
  specific time that conflict with the timeline. Short, vague answers do
  not trip it. The cloud run with a 7–8B model is where E2 and E4 become
  properly powered.
- Catching a liar is not the same as catching the murderer. In the run
  above both policies accused Rourke, whose greenhouse lie is the easiest
  to catch, while Vance is the culprit. That is a feature of the game and
  worth saying out loud rather than treating as a bug.

## Files added

| file | stage |
|---|---|
| `Deepseek.py` | shared API client, offline fallback |
| `Generator.py` | case generation, validation, repair loop |
| `Judge.py` | verdict from transcript + ground truth |
| `Experiments.py` | the four experiments, plots, report |
| `Test_Stages101112.py` | tests for all three |

---

# Game edition (stage 9 rewrite)

The frontend and `Api.py` were rebuilt around actual game mechanics. What
changed and why, point by point:

**Answers and scores are hidden until you commit.** `/api/explore` now
returns only the three questions and their kind. The answer and its band
stay on the server until `/api/commit`. This is enforced server-side, so
the UI cannot peek even if you edit it. Hard mode hides the kinds too.

**The case file is a document.** 📁 in the top bar opens it any time:
brief, persons of interest, what was known at the outset, and an evidence
board that grows as you play. It never contains secrets or knows-lists.

**Verification is the judge.** Accusing sends the transcript and the
truth to `deepseek-reasoner`, which reports whether you named the right
person AND whether you earned it. Runs as a background job because the
reasoner takes a minute or two; the UI polls and shows progress.

**DeepSeek is wired in for both jobs.** "new case" → generate with
DeepSeek runs `Generator.CaseGenerator` with the validator and repair
loop, saves to `cases/`, and loads it. Both fall back offline without a
key and stamp the result.

**Game mechanics.** Turn budget, score, three difficulties, evidence
board, per-suspect dossiers with pinned contradictions, composure bars,
lab tools that cost a turn. See `MANUAL.md` — also served at `/api/manual`
and behind the `?` button.

**Lab tools fixed.** Each tool now gets exactly the arguments it declares,
and the scene comes from the backend instead of a hard-coded string.
`Test_Api.py` calls all five with the UI's argument shapes so they cannot
drift apart again.

## Files

| file | purpose |
|---|---|
| `Api.py` | rewritten — game rules, case library, jobs, judge |
| `MANUAL.md` | the player manual |
| `frontend/src/App.jsx` | the interrogation room |
| `frontend/src/Panels.jsx` | case file, dossier, verdict, new case, help, the machine |
| `frontend/src/api.js` | fetch helper and job polling |
| `frontend/src/style.css` | noir theme |
| `Test_Api.py` | rewritten for the game mechanics |

## Scoring

```
+100  contradiction caught        -10  each turn
 +50  lab confirms a contradiction -15  extra for a wasted turn
 +25  new fact on the board       +500 / +150 / -200  accusation
```

## Running

```
set DEEPSEEK_API_KEY=sk-...           (or setx, then reopen the shell)
cd frontend && npm run build && cd ..
py Api.py
```

Open http://127.0.0.1:8000. The judge and generator both say "offline" in
the top bar if the key is not set.