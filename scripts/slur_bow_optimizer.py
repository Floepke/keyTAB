from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ui.dialogs.slur_bow_optimizer import (  # noqa: E402
    main,
    optimize_file,
    optimize_slur_bows_in_score,
)


if __name__ == "__main__":
    main()
