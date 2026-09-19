"""图片工具：读取 PNG / JPEG 尺寸（不引入 Pillow）。"""

from __future__ import annotations

import struct
from pathlib import Path

_JPEG_SOF = (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF)


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as fh:
            head = fh.read(26)
            if head.startswith(b"\x89PNG"):
                w, h = struct.unpack(">II", head[16:24])
                return w, h
            if head[:2] != b"\xff\xd8":
                return None
            fh.seek(2)
            while True:
                marker = fh.read(2)
                if len(marker) < 2 or marker[0] != 0xFF:
                    return None
                if marker[1] in _JPEG_SOF:
                    fh.read(3)
                    h, w = struct.unpack(">HH", fh.read(4))
                    return w, h
                (seg_len,) = struct.unpack(">H", fh.read(2))
                fh.seek(seg_len - 2, 1)
    except (OSError, struct.error):
        return None
