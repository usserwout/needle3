import re

import pytest

from conftest import requires_engine
from needle import environments

CATEGORIES = {"positive", "missing", "irrelevant", "negation", "invalid", "parallel"}


@pytest.fixture(params=sorted(environments.ENVIRONMENTS))
def env(request):
    return environments.ENVIRONMENTS[request.param]


def _schemas(env):
    return {fn._needle_tool["name"]: fn._needle_tool for fn in env.TOOLS}


def test_registry_matches_modules():
    for name, module in environments.ENVIRONMENTS.items():
        assert module.__name__ == f"needle.environments.{name}"
        assert module.SYSTEM and module.TOOLS and module.TEST_CASES


def test_tool_surface(env):
    schemas = _schemas(env)
    assert 0 < len(schemas) <= 5
    for schema in schemas.values():
        assert schema["description"]
        for prop in schema["parameters"]["properties"].values():
            assert "type" in prop


def test_cases_cover_all_categories(env):
    assert {case["category"] for case in env.TEST_CASES} == CATEGORIES


def test_cases_match_declared_tools(env):
    schemas = _schemas(env)
    for case in env.TEST_CASES:
        for call in case["calls"]:
            parameters = schemas[call["name"]]["parameters"]
            arguments = call["arguments"]
            assert set(arguments) <= set(parameters["properties"])
            assert set(parameters.get("required", [])) <= set(arguments)
            for key, value in arguments.items():
                prop = parameters["properties"][key]
                if "enum" in prop:
                    assert value in prop["enum"]
                if "minimum" in prop:
                    assert value >= prop["minimum"]
                if "maximum" in prop:
                    assert value <= prop["maximum"]
                if "pattern" in prop:
                    assert re.match(prop["pattern"], value)


def test_tools_execute(env):
    schemas = {fn._needle_tool["name"]: fn for fn in env.TOOLS}
    for case in env.TEST_CASES:
        for call in case["calls"]:
            result = schemas[call["name"]](**call["arguments"])
            assert result["ok"] is True


@requires_engine
def test_smart_home_smoke():
    from needle.environments import smart_home

    response = smart_home.agent.complete("turn on the kitchen lights")
    assert response.get("type") in ("call", "text", "refuse")
