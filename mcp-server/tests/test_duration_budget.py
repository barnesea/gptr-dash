from pathlib import Path
import importlib.util
import sys


SERVER_PATH = Path(__file__).resolve().parents[1] / "server.py"
sys.path.insert(0, str(SERVER_PATH.parent))
SPEC = importlib.util.spec_from_file_location("gptr_mcp_server_duration", SERVER_PATH)
server = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(server)


def test_duration_validation_and_default_public_signature():
    duration, error = server.validate_research_duration_input(60)
    assert duration == 60
    assert error is None
    assert server.deep_research.__defaults__[-1] == 60


def test_duration_validation_returns_structured_error():
    duration, error = server.validate_research_duration_input(601)
    assert duration is None
    assert error["status"] == "invalid_research_duration"
    assert error["min_research_duration_seconds"] == 15
    assert error["max_research_duration_seconds"] == 600
