"""Download-independent preprocessing for the Kaggle architecture study.

The setup script downloads the official sources, then invokes this module to
convert each source into the small JSONL contract consumed by ``needle.study``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _read_json(path: Path) -> Any:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        # The official SNIPS PlayMusic training file contains one emoji encoded
        # as a CESU-8 surrogate pair. Preserve it as the intended Unicode code
        # point rather than discarding the example or inserting replacement
        # characters.
        text = raw.decode("utf-8", errors="surrogatepass")
        text = text.encode("utf-16", errors="surrogatepass").decode("utf-16")
    return json.loads(text)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        raise ValueError(f"preprocessing produced no rows for {path}")
    print(f"  {path}: {count:,} rows")
    return count


def _droidcall(path: Path) -> Iterable[dict[str, Any]]:
    for row in _read_jsonl(path):
        if row.get("query") and row.get("tools") and row.get("answers"):
            yield {**row, "domain": "droidcall", "is_language_sample": False}


def _mobile_actions(path: Path, split: str) -> Iterable[dict[str, Any]]:
    for row in _read_jsonl(path):
        if row.get("metadata") != split:
            continue
        messages = row.get("messages") or []
        user = next((m for m in messages if m.get("role") == "user"), None)
        assistant = next(
            (m for m in reversed(messages)
             if m.get("role") == "assistant" and m.get("tool_calls")),
            None,
        )
        if not user or not assistant:
            continue
        answers = []
        for index, call in enumerate(assistant["tool_calls"]):
            function = call.get("function") or {}
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments or "{}")
            answers.append({
                "id": index,
                "name": function.get("name", ""),
                "arguments": arguments,
            })
        tools = [item.get("function", item) for item in row.get("tools", [])]
        if answers and tools:
            system = "\n".join(
                str(m.get("content") or "")
                for m in messages
                if m.get("role") in {"system", "developer"}
            ).strip()
            yield {
                "domain": "mobile_actions",
                "query": user.get("content", ""),
                "system": system,
                "tools": tools,
                "answers": answers,
                "is_language_sample": False,
            }


def _snips(root: Path, split: str) -> Iterable[dict[str, Any]]:
    base = root / "2017-06-custom-intent-engines"
    pattern = "train_*_full.json" if split == "train" else "validate_*.json"
    sources = sorted(base.glob(f"*/{pattern}"))
    if not sources:
        raise FileNotFoundError(f"no SNIPS {split} files found below {base}")

    parsed: list[tuple[str, list[dict[str, Any]]]] = []
    slots_by_intent: dict[str, set[str]] = {}
    for source in sources:
        payload = _read_json(source)
        for intent, samples in payload.items():
            parsed.append((intent, samples))
            slots = slots_by_intent.setdefault(intent, set())
            for sample in samples:
                for part in sample.get("data", []):
                    slot = part.get("slot_name") or part.get("entity")
                    if slot:
                        slots.add(slot)

    tools = [{
        "name": intent,
        "description": f"Handle the {intent} intent.",
        "parameters": {
            "type": "object",
            "properties": {
                slot: {"type": "string"}
                for slot in sorted(slots_by_intent[intent])
            },
        },
    } for intent in sorted(slots_by_intent)]

    for intent, samples in parsed:
        for sample in samples:
            parts = sample.get("data", [])
            query = "".join(str(part.get("text", "")) for part in parts).strip()
            arguments: dict[str, Any] = {}
            for part in parts:
                slot = part.get("slot_name") or part.get("entity")
                value = str(part.get("text", "")).strip()
                if slot and value:
                    if slot in arguments:
                        current = arguments[slot]
                        arguments[slot] = current + [value] if isinstance(current, list) else [current, value]
                    else:
                        arguments[slot] = value
            if query:
                yield {
                    "domain": "snips",
                    "query": query,
                    "tools": tools,
                    "answers": [{"id": 0, "name": intent, "arguments": arguments}],
                    "is_language_sample": False,
                }


def _dstc8_tools(schema: dict[str, Any]) -> list[dict[str, Any]]:
    slot_map = {slot["name"]: slot for slot in schema.get("slots", [])}
    tools = []
    for intent in schema.get("intents", []):
        slot_names = list(intent.get("required_slots", []))
        slot_names.extend(intent.get("optional_slots", {}).keys())
        properties = {}
        for name in dict.fromkeys(slot_names):
            slot = slot_map.get(name, {})
            prop: dict[str, Any] = {
                "type": "string",
                "description": slot.get("description", ""),
            }
            values = slot.get("possible_values") or []
            if slot.get("is_categorical") and values:
                prop["enum"] = values
            properties[name] = prop
        tools.append({
            "name": intent["name"],
            "description": intent.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": intent.get("required_slots", []),
            },
        })
    return tools


def _dstc8(root: Path, split: str) -> Iterable[dict[str, Any]]:
    split_dir = root / split
    schemas = {
        item["service_name"]: item
        for item in json.loads((split_dir / "schema.json").read_text(encoding="utf-8"))
    }
    tools = {name: _dstc8_tools(schema) for name, schema in schemas.items()}
    files = sorted(split_dir.glob("dialogues_*.json"))
    if not files:
        raise FileNotFoundError(f"no DSTC8 dialogue files found below {split_dir}")

    for source in files:
        for dialogue in json.loads(source.read_text(encoding="utf-8")):
            history: list[str] = []
            for turn in dialogue.get("turns", []):
                speaker = turn.get("speaker", "")
                utterance = str(turn.get("utterance", "")).strip()
                if utterance:
                    history.append(f"{speaker}: {utterance}")
                if speaker != "USER":
                    continue
                query = "\n".join(history)
                for frame_index, frame in enumerate(turn.get("frames", [])):
                    state = frame.get("state") or {}
                    intent = state.get("active_intent")
                    service = frame.get("service")
                    if not intent or intent == "NONE" or service not in tools:
                        continue
                    arguments = {
                        name: values[0] if len(values) == 1 else values
                        for name, values in (state.get("slot_values") or {}).items()
                        if values
                    }
                    yield {
                        "domain": "dstc8",
                        "query": query,
                        "tools": tools[service],
                        "answers": [{
                            "id": frame_index,
                            "name": intent,
                            "arguments": arguments,
                        }],
                        "is_language_sample": False,
                    }


def _fineweb(path: Path) -> Iterable[dict[str, Any]]:
    for row in _read_jsonl(path):
        text = str(row.get("text", "")).strip()
        if text:
            yield {
                "domain": "fineweb_edu",
                "text": text,
                "is_language_sample": True,
            }


def prepare_kaggle_data(root: Path, sample_size: int = 1000) -> None:
    raw = root / "data" / "raw"
    normalized = root / "data" / "normalized"
    print("Preprocessing official study datasets...")
    _write_jsonl(normalized / "droidcall_train.jsonl", _droidcall(raw / "droidcall" / "DroidCall_train.jsonl"))
    _write_jsonl(normalized / "droidcall_eval.jsonl", _droidcall(raw / "droidcall" / "DroidCall_test.jsonl"))
    _write_jsonl(normalized / "mobile_actions_train.jsonl", _mobile_actions(raw / "mobile_actions" / "dataset.jsonl", "train"))
    _write_jsonl(normalized / "mobile_actions_eval.jsonl", _mobile_actions(raw / "mobile_actions" / "dataset.jsonl", "eval"))
    _write_jsonl(normalized / "snips_train.jsonl", _snips(raw / "snips", "train"))
    _write_jsonl(normalized / "snips_eval.jsonl", _snips(raw / "snips", "eval"))
    _write_jsonl(normalized / "dstc8_train.jsonl", _dstc8(raw / "dstc8", "train"))
    _write_jsonl(normalized / "dstc8_eval.jsonl", _dstc8(raw / "dstc8", "dev"))
    _write_jsonl(normalized / "fineweb_edu_train.jsonl", _fineweb(raw / "fineweb_edu" / "train.jsonl"))

    config = {
        "base_checkpoint": "checkpoints/needle3.safetensors",
        "output_dir": "study_runs",
        "sample_size": sample_size,
        "seed": 20260921,
        "seq_len": 512,
        "dataset_paths": {
            name: f"data/normalized/{name}_train.jsonl"
            for name in ("droidcall", "mobile_actions", "snips", "dstc8", "fineweb_edu")
        },
        "evaluation_paths": [
            f"data/normalized/{name}_eval.jsonl"
            for name in ("droidcall", "mobile_actions", "snips", "dstc8")
        ],
    }
    (root / "study.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print("  study.yaml written")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize official Needle study datasets")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--sample-size", type=int, default=1000)
    args = parser.parse_args()
    prepare_kaggle_data(args.root.resolve(), args.sample_size)


if __name__ == "__main__":
    main()
