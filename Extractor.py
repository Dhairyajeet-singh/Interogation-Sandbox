"""
extractor.py   -  Stage 3b
--------------------------
Reads a suspect's answer and returns which known case facts it mentions.

This is plain string matching against the alias lists in the case file.
No model is involved, which is the whole point: the question scores in
stage 5 must be countable and reproducible, not another AI's opinion.

Anything that looks like it should have matched but did not is written
to a miss log, so the alias lists can be improved over time.
"""

import re
from collections import Counter as _Counter


def normalise(text):
    """
    Lowercase, strip punctuation, collapse whitespace.

    "The GREENHOUSE, out back." -> "the greenhouse out back"

    Times keep their colon, because "21:38" must not become "2138".
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9:\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class FactExtractor:
    def __init__(self, entities):
        """
        entities: the "entities" dict from the case file.

        Builds one flat list of (normalised_alias, entity_id), sorted
        longest first. Longest-first matters: "brass counterweight"
        must win over "counterweight" so we do not count it twice.
        """
        self.entities = entities
        self.alias_list = []
        seen = {}

        for ent_id, ent in entities.items():
            for alias in ent["aliases"]:
                key = normalise(alias)
                if not key:
                    continue
                if key in seen and seen[key] != ent_id:
                    raise ValueError(
                        f"alias {alias!r} is ambiguous: "
                        f"maps to both {seen[key]} and {ent_id}"
                    )
                seen[key] = ent_id
                self.alias_list.append((key, ent_id))

        self.alias_list.sort(key=lambda pair: len(pair[0]), reverse=True)
        self.misses = _Counter()

    def extract(self, text):
        """
        Return the set of entity ids mentioned in `text`.

        Matched spans are blanked out as they are found, so overlapping
        aliases cannot both fire on the same words.
        """
        haystack = " " + normalise(text) + " "
        found = set()

        for alias, ent_id in self.alias_list:
            needle = " " + alias + " "
            if needle in haystack:
                found.add(ent_id)
                haystack = haystack.replace(needle, " " + "\u0000" * len(alias) + " ")

        self._log_misses(haystack)
        return found

    def extract_by_type(self, text):
        """Same as extract(), grouped by entity type."""
        out = {}
        for ent_id in self.extract(text):
            t = self.entities[ent_id]["type"]
            out.setdefault(t, set()).add(ent_id)
        return out

    def verifiable_claims(self, text):
        """
        Entities the timeline graph could adjudicate: a time, a place,
        or a person. "I was in the greenhouse at nine" makes two.
        "I don't recall" makes none.
        """
        keep = {"time", "place", "person"}
        return {e for e in self.extract(text)
                if self.entities[e]["type"] in keep}

    def _log_misses(self, leftover):
        """
        Record capitalised-looking or time-looking phrases that matched
        nothing, so alias lists can be grown. Advisory only.
        """
        for token in re.findall(r"\b\d{1,2}:\d{2}\b", leftover):
            self.misses[token] += 1

    def miss_report(self, top=10):
        return self.misses.most_common(top)


def load_extractor(case):
    """Convenience: build an extractor straight from a loaded case dict."""
    return FactExtractor(case["entities"])