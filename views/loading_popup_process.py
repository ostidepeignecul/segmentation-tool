from __future__ import annotations

import argparse
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

try:
    from views.loading_popup_view import LoadingPopupView
except ImportError:
    from loading_popup_view import LoadingPopupView


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Standalone loading popup.")
    parser.add_argument("--title", default="Loading")
    parser.add_argument("--message", default="Operation in progress...")
    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    popup = LoadingPopupView(
        title=str(args.title or "Loading"),
        message=str(args.message or "Operation in progress..."),
        show_log=False,
    )
    popup.setModal(False)
    popup.setWindowModality(Qt.WindowModality.NonModal)
    popup.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    popup.show()
    popup.raise_()
    popup.activateWindow()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
