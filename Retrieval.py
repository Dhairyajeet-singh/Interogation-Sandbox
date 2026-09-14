"""
retrieval.py   -  Stage 3
-------------------------
Finds the case facts relevant to a question, three different ways, then
filters by what the suspect is actually allowed to know.

  1. keyword search  (BM25)         - good for exact words: "counterweight"
  2. meaning search  (embeddings)   - good for vague words: "what did you see"
  3. timeline graph  (NetworkX)     - good for "where was X at nine"

The three result lists are merged with Reciprocal Rank Fusion, which is
just: an item ranked highly by several methods beats an item ranked
highly by one.

The knowledge filter runs LAST and is the important part. A suspect can
only ever be handed facts in their own knows-list, so they physically
cannot cite evidence they never saw.

Embeddings are optional. If sentence-transformers is not installed the
system runs with BM25 + graph only, and says so.
"""

import re

import networkx as nx
from rank_bm25 import BM25Okapi

try:
    from sentence_transformers import SentenceTransformer, util
    HAVE_EMBEDDINGS = True
except ImportError:
    HAVE_EMBEDDINGS = False


def tokenise(text):
    """Split into lowercase words. BM25 needs a list of tokens."""
    return re.findall(r"[a-z0-9:]+", text.lower())


# ======================================================================
# timeline graph
# ======================================================================

class Timeline:
    """
    The case timeline as a graph.

    Nodes are people, places and times. An edge person -> place means
    that person claims to have been there, labelled with the window.

    Used to answer: was this person in this place at this time, and does
    a stated alibi fit the record.
    """

    def __init__(self, entries):
        self.entries = entries
        self.graph = nx.MultiDiGraph()
        for e in entries:
            self.graph.add_node(e["actor"], kind="person")
            self.graph.add_node(e["place"], kind="place")
            self.graph.add_edge(
                e["actor"], e["place"],
                start=e["from"], end=e["to"],
                verifiable_by=e["verifiable_by"],
            )

    @staticmethod
    def _minutes(hhmm):
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)

    def where_was(self, actor_id):
        """All recorded placements for one person."""
        return [e for e in self.entries if e["actor"] == actor_id]

    def who_was_at(self, place_id, at=None):
        """Everyone recorded at a place, optionally at a given time."""
        out = []
        for e in self.entries:
            if e["place"] != place_id:
                continue
            if at is None:
                out.append(e)
            elif self._minutes(e["from"]) <= self._minutes(at) <= self._minutes(e["to"]):
                out.append(e)
        return out

    def check_alibi(self, actor_id, place_id, at):
        """
        Does the record put this person in this place at this time?

        Returns (verdict, explanation) where verdict is
        "supported" | "contradicted" | "unknown".
        """
        placements = self.where_was(actor_id)
        if not placements:
            return "unknown", f"no record for {actor_id}"

        t = self._minutes(at)
        for e in placements:
            if self._minutes(e["from"]) <= t <= self._minutes(e["to"]):
                if e["place"] == place_id:
                    return "supported", (
                        f"record puts {actor_id} at {place_id} "
                        f"{e['from']}-{e['to']}"
                    )
                return "contradicted", (
                    f"record puts {actor_id} at {e['place']} "
                    f"({e['from']}-{e['to']}), not {place_id}"
                )
        return "unknown", f"no record for {actor_id} at {at}"

    def facts_about(self, entity_ids):
        """
        Turn graph edges touching these entities into readable lines.
        This is the graph's contribution to retrieved context.
        """
        lines = []
        for e in self.entries:
            if e["actor"] in entity_ids or e["place"] in entity_ids:
                line = f"{e['actor']} was at {e['place']} from {e['from']} to {e['to']}"
                if e["verifiable_by"]:
                    line += f" (recorded by {e['verifiable_by']})"
                lines.append(line)
        return lines


# ======================================================================
# hybrid search
# ======================================================================

class HybridRetriever:
    """
    BM25 + optional embeddings + timeline graph, fused with RRF.
    """

    def __init__(self, entities, timeline, embed_model="BAAI/bge-small-en-v1.5"):
        self.entities = entities
        self.timeline = timeline

        # one searchable document per entity: its text plus its aliases
        self.ids = list(entities.keys())
        self.docs = [
            entities[i]["text"] + " " + " ".join(entities[i]["aliases"])
            for i in self.ids
        ]

        self.bm25 = BM25Okapi([tokenise(d) for d in self.docs])

        self.embedder = None
        self.doc_vecs = None
        if HAVE_EMBEDDINGS:
            self.embedder = SentenceTransformer(embed_model, device="cpu")
            self.doc_vecs = self.embedder.encode(
                self.docs, convert_to_tensor=True, show_progress_bar=False
            )

    # -------------------------------------------------- individual signals
    def search_keyword(self, query, k=8):
        """BM25. Returns entity ids best first."""
        scores = self.bm25.get_scores(tokenise(query))
        ranked = sorted(zip(self.ids, scores), key=lambda p: p[1], reverse=True)
        return [eid for eid, s in ranked[:k] if s > 0]

    def search_meaning(self, query, k=8):
        """Embeddings. Returns [] if sentence-transformers is absent."""
        if self.embedder is None:
            return []
        qv = self.embedder.encode(query, convert_to_tensor=True,
                                  show_progress_bar=False)
        sims = util.cos_sim(qv, self.doc_vecs)[0]
        order = sims.argsort(descending=True)[:k]
        return [self.ids[i] for i in order.tolist()]

    def search_graph(self, query, k=8):
        """
        Timeline lookup. Any person, place or time named in the query
        pulls in the entities connected to it in the graph.
        """
        named = []
        low = query.lower()
        for eid, ent in self.entities.items():
            if ent["type"] not in ("person", "place", "time"):
                continue
            for alias in ent["aliases"]:
                if alias.lower() in low:
                    named.append(eid)
                    break

        out = []
        for e in self.timeline.entries:
            if e["actor"] in named or e["place"] in named:
                for eid in (e["actor"], e["place"]):
                    if eid not in out:
                        out.append(eid)
                if e["verifiable_by"] and e["verifiable_by"] not in out:
                    out.append(e["verifiable_by"])
        return out[:k]

    # -------------------------------------------------- fusion
    @staticmethod
    def fuse(rankings, k=60):
        """
        Reciprocal Rank Fusion.

        Each list votes. An item at rank r contributes 1/(k+r).
        k=60 is the standard constant from the original paper; it stops
        the top rank from dominating everything below it.
        """
        scores = {}
        for ranking in rankings:
            for rank, item in enumerate(ranking):
                scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=scores.get, reverse=True)

    # -------------------------------------------------- the public call
    def retrieve(self, query, allowed_ids, top_k=4):
        """
        Find facts for this question that this suspect is allowed to know.

        allowed_ids is the suspect's knows-list. The filter runs last and
        is absolute: nothing outside it can ever be returned.
        """
        fused = self.fuse([
            self.search_keyword(query),
            self.search_meaning(query),
            self.search_graph(query),
        ])
        permitted = [eid for eid in fused if eid in allowed_ids]
        return permitted[:top_k]

    def as_text(self, entity_ids):
        """Render retrieved entities as lines for the prompt."""
        return [self.entities[e]["text"] for e in entity_ids]


# ======================================================================
# knowledge filter / violation check
# ======================================================================

class KnowledgeChecker:
    """
    Catches a suspect citing something outside their knows-list.

    Public facts are known by everyone, so they are always permitted.
    """

    def __init__(self, extractor, public_facts):
        self.extractor = extractor
        self.public = set(public_facts)

    def violations(self, answer, allowed_ids):
        """Entity ids the answer mentions that this suspect should not know."""
        mentioned = self.extractor.extract(answer)
        permitted = self.public | set(allowed_ids)
        return mentioned - permitted