import json
from pathlib import Path

from needle.study.kaggle_data import _droidcall, _dstc8, _mobile_actions, _read_json, _snips


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_droidcall_keeps_official_tool_call_contract(tmp_path):
    source = tmp_path / "droid.jsonl"
    row = {"query": "call Sam", "tools": [{"name": "call"}], "answers": [{"name": "call", "arguments": {"name": "Sam"}}]}
    _write_jsonl(source, [row])
    assert list(_droidcall(source))[0]["query"] == "call Sam"


def test_mobile_actions_uses_metadata_split_and_decodes_arguments(tmp_path):
    source = tmp_path / "mobile.jsonl"
    _write_jsonl(source, [{
        "metadata": "train",
        "tools": [{"function": {"name": "call"}}],
        "messages": [
            {"role": "user", "content": "call Sam"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "call", "arguments": "{\"name\": \"Sam\"}"}}]},
        ],
    }])
    rows = list(_mobile_actions(source, "train"))
    assert rows[0]["answers"][0]["arguments"] == {"name": "Sam"}
    assert list(_mobile_actions(source, "eval")) == []


def test_snips_reads_official_train_and_validate_layout(tmp_path):
    intent_dir = tmp_path / "2017-06-custom-intent-engines" / "PlayMusic"
    intent_dir.mkdir(parents=True)
    payload = {"PlayMusic": [{"data": [{"text": "play "}, {"text": "Halo", "entity": "track"}]}]}
    (intent_dir / "train_PlayMusic_full.json").write_text(json.dumps(payload), encoding="utf-8")
    (intent_dir / "validate_PlayMusic.json").write_text(json.dumps(payload), encoding="utf-8")
    train = list(_snips(tmp_path, "train"))
    valid = list(_snips(tmp_path, "eval"))
    assert train[0]["query"] == "play Halo"
    assert train[0]["answers"][0]["arguments"] == {"track": "Halo"}
    assert valid[0]["answers"][0]["name"] == "PlayMusic"


def test_snips_legacy_cesu8_emoji_is_preserved(tmp_path):
    source = tmp_path / "play_music.json"
    source.write_bytes(
        b'{"PlayMusic":[{"data":[{"text":"Pop Punk Perfection '
        + b"\xed\xa0\xbc\xed\xbd\x95"
        + b'"}]}]}'
    )
    payload = _read_json(source)
    assert payload["PlayMusic"][0]["data"][0]["text"] == "Pop Punk Perfection \U0001f355"


def test_dstc8_converts_user_state_with_dialogue_context(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    schema = [{
        "service_name": "Music_1",
        "slots": [{"name": "track", "description": "Track name", "is_categorical": False}],
        "intents": [{
            "name": "PlayMusic",
            "description": "Play music",
            "required_slots": ["track"],
            "optional_slots": {},
        }],
    }]
    (train / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    dialogues = [{
        "turns": [{
            "speaker": "USER",
            "utterance": "Play Halo",
            "frames": [{
                "service": "Music_1",
                "state": {
                    "active_intent": "PlayMusic",
                    "slot_values": {"track": ["Halo"]},
                },
            }],
        }],
    }]
    (train / "dialogues_001.json").write_text(json.dumps(dialogues), encoding="utf-8")
    row = list(_dstc8(tmp_path, "train"))[0]
    assert row["query"] == "USER: Play Halo"
    assert row["answers"][0]["arguments"] == {"track": "Halo"}
