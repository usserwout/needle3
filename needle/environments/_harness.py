import json

import needle

_agents = {}


def agent_for(module):
    key = module.__name__
    if key not in _agents:
        _agents[key] = needle.Needle(tools=module.TOOLS, system=module.SYSTEM)
    return _agents[key]


def run_tests(module, min_confidence=0.0, verbose=True):
    """Run an environment's frozen acceptance suite against the shipped engine.
    The default scores raw model output; passing e.g. min_confidence=0.4 applies
    the production contract (act on a call only at or above the threshold,
    otherwise treat it as a refusal)."""
    agent = agent_for(module)
    failures, critical_failures = [], []
    for case in module.TEST_CASES:
        agent.reset()
        response = agent.complete(case["query"])
        got = response.get("function_calls") or []
        validation = response.get("validation") or {}
        if got and (validation.get("ungrounded") or validation.get("negation")):
            got = []
        if got and response.get("confidence", 0.0) < min_confidence:
            got = []
        want = case["calls"]
        ok = got == want or sorted(
            json.dumps(c, sort_keys=True) for c in got
        ) == sorted(json.dumps(c, sort_keys=True) for c in want)
        if not ok:
            failures.append(case)
            if case.get("critical"):
                critical_failures.append(case)
            if verbose:
                print(f"FAIL [{case['category']}] {case['query']}")
                print(f"  want {json.dumps(want)}")
                print(f"  got  {json.dumps(got)}")
    passed = len(module.TEST_CASES) - len(failures)
    print(f"{passed}/{len(module.TEST_CASES)} passed, {len(critical_failures)} critical failures "
          f"(confidence gate {min_confidence})")
    return passed >= round(0.9 * len(module.TEST_CASES)) and not critical_failures
