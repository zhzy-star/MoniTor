"""Run repository entry points with ``python -m src.<module>``."""
import sys
from pathlib import Path

# ImageBind uses absolute imports internally. Resolve its checkout from this
# file so invocation does not depend on the shell's working directory.
_imagebind_root = str(Path(__file__).resolve().parents[1] / "libs" / "ImageBind")
if _imagebind_root not in sys.path:
    sys.path.append(_imagebind_root)
