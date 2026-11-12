"""Verify Qt dropEvent path extraction (including Unicode paths) without a mouse drag.

Builds a real QDropEvent / QMimeData and delivers it to the live MainWindow.
Run: python tests/test_drop.py
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import stem3  # noqa: E402

from PySide6.QtCore import QMimeData, QPointF, Qt, QTimer, QUrl  # noqa: E402
from PySide6.QtGui import QDropEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402


def main():
    tmp = Path(tempfile.mkdtemp(prefix="stem3_drop_"))
    plain = tmp / "song.wav"
    unicode_name = tmp / "测试 音乐 中文.wav"
    try:
        plain.write_bytes(b"")
        unicode_name.write_bytes(b"")  # only path transfer is tested
        expected = [str(plain), str(unicode_name)]

        cfg = argparse.Namespace(mode=3, outdir="", whole=False, threads=4, nogui=False)
        app = QApplication.instance() or QApplication(sys.argv)
        captured = []

        def inject():
            wins = [w for w in app.topLevelWidgets()
                    if isinstance(w, QWidget) and w.windowTitle().startswith("音轨分离")]
            if not wins:
                print("NO WINDOW FOUND")
                app.quit()
                return
            win = wins[0]
            win.start_files = lambda files: captured.append(list(files))
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(p) for p in expected])
            event = QDropEvent(QPointF(30.0, 30.0), Qt.CopyAction, mime,
                               Qt.LeftButton, Qt.NoModifier)
            win.dropEvent(event)
            app.quit()

        QTimer.singleShot(800, inject)
        stem3.run_qt_gui([], cfg)  # blocks until app.quit()

        got = captured[0] if captured else []
        # QUrl.toLocalFile() returns forward slashes on Windows; compare as paths.
        ok = bool(got) and [Path(f) for f in got] == [Path(f) for f in expected]
        print("expected:", expected)
        print("received:", got)
        print("MATCH:", ok)
        if not ok:
            raise SystemExit(1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
