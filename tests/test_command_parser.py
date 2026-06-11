# _max_cyan_ — project_mxsa
import pytest
from simba.voice.command_parser import CommandParser

def test_exact_match():
    parser = CommandParser()
    result = parser.parse("fetch the bottle")
    assert result is not None
    assert result["action"] == "fetch"
    assert result["target"] == "bottle"

def test_fuzzy_match():
    parser = CommandParser()
    # "bringe me the" is a typo for "bring me the", target extraction might fail on fuzzy match
    result = parser.parse("bringe me the remote")
    assert result is not None
    assert result["action"] == "fetch"

def test_fillers_removed():
    parser = CommandParser()
    result = parser.parse("um simba could you please fetch the bottle quickly")
    assert result is not None
    assert result["action"] == "fetch"
    assert result["target"] == "bottle"

def test_no_match():
    parser = CommandParser()
    result = parser.parse("qwerty uiop asdfg")
    assert result is None

def test_extract_target():
    parser = CommandParser()
    result = parser.parse("grab the blue ball")
    assert result is not None
    assert result["action"] == "grab"
    assert result["target"] == "blue ball"

def test_identity_commands():
    parser = CommandParser()
    result = parser.parse("who are you")
    assert result is not None
    assert result["action"] == "who_are_you"
    assert result["target"] is None
