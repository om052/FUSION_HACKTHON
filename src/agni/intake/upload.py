"""Safe temporary-file handling for uploaded, untrusted media.

Uploaded files are untrusted input.  This module writes an upload
to a private temporary file (never persisting it in the project
tree or any user-visible location) and guarantees cleanup, even
when the consumer raises.

The file is created with ``delete=False`` so it can be reopened by
the decoding stack (OpenCV / FFmpeg) by path, then removed in the
context manager's ``finally`` block.
"""

from contextlib import contextmanager
from pathlib import Path
import tempfile
from typing import Iterator, Optional


@contextmanager
def temporary_upload(
    data: bytes,
    suffix: str = "",
    prefix: str = "agni_upload_",
) -> Iterator[Path]:
    """Write upload bytes to a temporary file and clean it up.

    The file is written to the process temporary directory with a
    randomized name, so it is never persisted in the application
    tree and cannot be addressed by a user-supplied path.

    Args:
        data: Raw uploaded bytes.
        suffix: Desired file extension (e.g. ``".mp4"``) so the
            decoder can sniff the container.
        prefix: Temporary filename prefix.

    Yields:
        The absolute path of the temporary file.  The file is
        deleted when the context exits, whether or not the consumer
        raised.
    """
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=prefix, suffix=suffix, delete=False
        ) as handle:
            handle.write(data)
            temp_path = Path(handle.name)
        yield temp_path
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
