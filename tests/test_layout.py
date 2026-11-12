"""Verify no QLabel is squashed by layout and font sizes scale from the app base.

Run: python tests/test_layout.py
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import stem3  # noqa: E402

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QRadioButton  # noqa: E402


def main():
    cfg = argparse.Namespace(mode=3, outdir="", whole=False, threads=4, nogui=False)
    app = QApplication.instance() or QApplication(sys.argv)
    result = {}

    def scan(win):
        # A wordWrap QLabel's sizeHint uses a default width; heightForWidth()
        # is the height actually required at the widget's current width.
        bad = []
        for lab in win.findChildren(QLabel):
            need = lab.heightForWidth(lab.width())
            if need > 0 and lab.height() < need - 2:
                bad.append((lab.objectName() or "plain", lab.text()[:22],
                            lab.height(), need))
        return bad

    def check():
        wins = [w for w in app.topLevelWidgets()
                if w.windowTitle().startswith("音轨分离")]
        win = wins[0]
        app.processEvents()
        labels = win.findChildren(QLabel)
        lims = [l for l in labels if "libsndfile" in l.text()]
        lim = lims[0] if lims else None
        title = next((l for l in labels if l.objectName() == "title"), None)
        result.update(
            compressed=scan(win),
            lim_h=lim.height() if lim else -1,
            lim_need=lim.heightForWidth(lim.width()) if lim else -1,
            lim_divs=lim.text().count("<div") if lim else 0,
            base_pt=app.font().pointSizeF(),
            title_pt=title.font().pointSizeF() if title else -1,
            lim_pt=lim.font().pointSizeF() if lim else -1,
        )
        win.resize(610, 520)  # small / high-DPI screen
        app.processEvents()
        result["compressed_small"] = scan(win)
        result["radios"] = [
            (r.isChecked(), r.height()) for r in win.findChildren(QRadioButton)
        ]
        app.quit()

    QTimer.singleShot(900, check)
    stem3.run_qt_gui([], cfg)  # blocks until app.quit()

    print("font pt  base=%.2f  title=%.2f  limits=%.2f"
          % (result["base_pt"], result["title_pt"], result["lim_pt"]))
    print("limits   divs=%d  height=%d  needAtWidth=%d"
          % (result["lim_divs"], result["lim_h"], result["lim_need"]))
    print("compressed (normal)  :", result["compressed"])
    print("compressed (610x520) :", result["compressed_small"])
    print("radios               :", result["radios"])
    ok = (not result["compressed"]
          and not result["compressed_small"]
          and result["lim_divs"] == len(stem3.LIMIT_ROWS)
          and result["lim_h"] >= result["lim_need"] - 2
          and result["title_pt"] > result["base_pt"] > result["lim_pt"] > 0
          and sum(1 for checked, _ in result["radios"] if checked) == 1)
    print("LAYOUT OK:", ok)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
