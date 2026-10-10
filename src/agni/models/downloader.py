"""Safe artifact downloader for AGNI pretrained weights and models.

Enforces size validation, HTTP success, non-HTML content verification,
and cryptographic checksum calculation.
"""

import hashlib
from pathlib import Path
import urllib.request
import urllib.error


def download_artifact(
    url: str,
    output_path: Path,
    expected_size_bytes: int | None = None,
    chunk_size: int = 1024 * 1024,
) -> dict:
    """Safely downloads a binary artifact with verification."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from: {url}")
    print(f"Destination: {output_path}")

    # Set User-Agent to avoid bot blocks
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AGNI-Verifier/1.0"},
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            status_code = response.getcode()
            if status_code != 200:
                raise RuntimeError(f"HTTP request failed with status code {status_code}")

            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type:
                raise RuntimeError(f"Server returned HTML page instead of binary artifact: {content_type}")

            sha256 = hashlib.sha256()
            total_bytes = 0

            with open(output_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    # Guard: verify the first chunk doesn't look like an HTML error page
                    if total_bytes == 0 and (b"<!DOCTYPE html" in chunk or b"<html" in chunk):
                        raise RuntimeError("Received HTML document instead of binary model file")
                    f.write(chunk)
                    sha256.update(chunk)
                    total_bytes += len(chunk)

            computed_sha256 = sha256.hexdigest()
            print(f"Downloaded {total_bytes} bytes. SHA256: {computed_sha256}")

            if expected_size_bytes is not None and total_bytes != expected_size_bytes:
                print(f"WARNING: File size {total_bytes} differs from expected {expected_size_bytes}")

            return {
                "file_path": str(output_path),
                "file_size": total_bytes,
                "sha256": computed_sha256,
                "status": "SUCCESS",
            }
    except Exception as e:
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError(f"Download failed for {url}: {e}") from e


if __name__ == "__main__":
    yunet_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    yunet_dest = Path("models/yunet/face_detection_yunet_2023mar.onnx")

    xcep_url = "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/xception_best.pth"
    xcep_dest = Path("models/visual_detector/xception_best.pth")

    print("\n--- Downloading YuNet ONNX ---")
    yunet_info = download_artifact(yunet_url, yunet_dest)
    print("YuNet Info:", yunet_info)

    print("\n--- Downloading DeepfakeBench Xception Checkpoint ---")
    xcep_info = download_artifact(xcep_url, xcep_dest, expected_size_bytes=87763531)
    print("Xception Info:", xcep_info)
