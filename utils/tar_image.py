"""Stream images directly from uncompressed .tar archives.

MatrixCity (BoDai/MatrixCity on Hugging Face) ships images as uncompressed
.tar files, e.g. `street/train/small_city_road_horizon.tar` containing
`small_city_road_horizon/0000.png`. Extracting all of them doubles disk
usage (~85 GiB). Since uncompressed tars support cheap random access, this
module lets the dataset readers open an image path that does not exist on
disk by locating a sibling/ancestor `<dir>.tar` archive and reading the
member bytes directly.

Usage:
    from utils.tar_image import open_image, image_available

    image = open_image(path)        # like Image.open, with tar fallback
    if image_available(path): ...   # like os.path.exists, with tar fallback

Notes:
- Member index (name -> data offset) is built once per archive and cached
  in memory for the lifetime of the process.
- File handles are cached per archive; readers here are single-threaded.
- Returned PIL images are fully loaded (detached from the tar buffer).
"""

import io
import os
import tarfile

from PIL import Image

# tar_path -> {member_name: (data_offset, size)}
_index_cache = {}
# tar_path -> persistent binary file handle
_handle_cache = {}


def _build_index(tar_path):
    """Scan archive headers only (data blocks are seek-skipped, so this is
    fast even for multi-GB uncompressed tars)."""
    index = {}
    with tarfile.open(tar_path, "r:") as tf:
        for member in tf:
            if member.isfile():
                name = member.name
                if name.startswith("./"):
                    name = name[2:]
                index[name] = (member.offset_data, member.size)
    _index_cache[tar_path] = index
    return index


def _lookup_in_tar(tar_path, member_name):
    if tar_path not in _index_cache:
        _build_index(tar_path)
    return _index_cache[tar_path].get(member_name)


def _resolve(image_path):
    """Walk up from the (missing) image path looking for an ancestor
    directory `A` such that `A.tar` exists and contains the file as a
    member. Returns (tar_path, member_name, offset, size) or None.

    Two member-name conventions are tried for each candidate archive:
    `A.tar` may store entries prefixed with A's directory name
    (e.g. `small_city_road_horizon/0000.png`) or unprefixed (`0000.png`).
    """
    p = os.path.normpath(image_path)
    cur = os.path.dirname(p)
    while cur and cur != os.path.dirname(cur):
        tar_path = cur + ".tar"
        if os.path.exists(tar_path):
            candidates = [
                os.path.relpath(p, os.path.dirname(cur)).replace(os.sep, "/"),
                os.path.relpath(p, cur).replace(os.sep, "/"),
            ]
            for member in candidates:
                hit = _lookup_in_tar(tar_path, member)
                if hit is not None:
                    return tar_path, member, hit[0], hit[1]
        cur = os.path.dirname(cur)
    return None


def image_available(image_path):
    """True if the image exists on disk or can be streamed from a tar."""
    if os.path.exists(image_path):
        return True
    return _resolve(image_path) is not None


def open_image(image_path):
    """Open like PIL.Image.open; falls back to streaming from a sibling or
    ancestor `.tar` archive when the file is not extracted on disk."""
    if os.path.exists(image_path):
        return Image.open(image_path)

    resolved = _resolve(image_path)
    if resolved is None:
        raise FileNotFoundError(
            f"{image_path} (not on disk and no sibling/ancestor .tar contains it)"
        )

    tar_path, member, offset, size = resolved
    fh = _handle_cache.get(tar_path)
    if fh is None:
        fh = open(tar_path, "rb")
        _handle_cache[tar_path] = fh
    fh.seek(offset)
    data = fh.read(size)
    if len(data) != size:
        raise IOError(f"short read for {member} in {tar_path}: {len(data)} < {size}")

    img = Image.open(io.BytesIO(data))
    img.load()  # detach from the in-memory buffer
    return img
