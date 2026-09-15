"""
test_extractor.py  -  Stage 3b acceptance test

Twenty hand-labelled answers. The extractor must find the expected
entity ids in each. This is the gate for stage 3b.
"""

import json
from Extractor import FactExtractor, normalise

case = json.load(open("Case_Ashfield.json", encoding="utf-8"))
ex = FactExtractor(case["entities"])

# (answer text, entity ids that MUST be found)
CASES = [
    ("I was in the greenhouse repotting.",
     {"loc_greenhouse"}),

    ("I was down by the boathouse most of the evening.",
     {"loc_boathouse"}),

    ("She was struck with the brass counterweight.",
     {"ev_counterweight"}),

    ("The weight came off the telescope mount.",
     {"ev_counterweight"}),

    ("I went up to the observatory at 21:38.",
     {"loc_observatory", "time_2138"}),

    ("I found her just before ten and called straight away.",
     {"time_2200", "ev_call_2204"}),

    ("The rain started at 9:20 so I came inside.",
     {"time_2120"}),

    ("I don't recall anything about that.",
     set()),

    ("Dr Marsh and I had disagreed about the paper.",
     {"per_marsh", "ev_paper"}),

    ("Her notebook is not something I ever touched.",
     {"ev_notebook"}),

    ("There was a wet umbrella in the east hall.",
     {"ev_umbrella", "loc_east_hall"}),

    ("Vance was downstairs in the study all evening.",
     {"per_vance", "loc_study"}),

    ("I was in the hall from nine until nearly ten.",
     {"loc_east_hall", "time_2100", "time_2200"}),

    ("The keycard reader logs everyone who goes up.",
     {"ev_keycard"}),

    ("My aunt changed her will last month.",
     {"per_marsh", "ev_will"}),

    ("Claire was very upset when she came down.",
     {"per_halloway"}),

    ("Fuel has been going missing from the generator.",
     {"ev_fuel"}),

    ("There was a mark on the desk where something had been.",
     {"ev_dust_outline"}),

    ("I left at 21:46 and went back to the study.",
     {"time_2146", "loc_study"}),

    ("Nothing. I have nothing to say to you.",
     set()),
]


def run():
    failures = []
    for text, expected in CASES:
        got = ex.extract(text)
        missing = expected - got
        if missing:
            failures.append((text, expected, got, missing))

    for text, expected, got, missing in failures:
        print(f"\nFAIL: {text!r}")
        print(f"  expected to contain : {sorted(expected)}")
        print(f"  actually found      : {sorted(got)}")
        print(f"  missing             : {sorted(missing)}")

    print(f"\n{len(CASES) - len(failures)}/{len(CASES)} answers extracted correctly")

    # verifiable-claims check
    v = ex.verifiable_claims("I was in the greenhouse at nine o'clock.")
    assert "loc_greenhouse" in v and "time_2100" in v, v
    assert ex.verifiable_claims("I don't remember.") == set()
    print("verifiable_claims ok")

    assert not failures, f"{len(failures)} extraction failures"
    print("\nstage 3b PASSED")


if __name__ == "__main__":
    run()