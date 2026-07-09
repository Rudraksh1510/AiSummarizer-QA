import threading
import time

from src.pdfsummarizer.utils import extract_text_from_files


class DummyFile:
    def __init__(self, name):
        self.name = name


def test_extract_text_from_files_processes_files_concurrently(monkeypatch):
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_extract(file):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.2)
        with lock:
            active -= 1
        return f"content from {file.name}"

    monkeypatch.setattr(
        "src.pdfsummarizer.utils._extract_text_from_single_file",
        fake_extract,
    )

    files = [DummyFile("a.txt"), DummyFile("b.txt")]
    chunks = extract_text_from_files(files)

    assert len(chunks) >= 1
    assert max_active >= 2
