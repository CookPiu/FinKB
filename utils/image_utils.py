"""图片工具：读取 PNG / JPEG 尺寸（不引入 Pillow）。"""
import struct

# JPEG 的 SOF 段标记，段内含图片宽高
_JPEG_SOF = (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF)


def image_size(path):
    """
    读取图片宽高
    :param path: 图片路径（Path）
    :return: (宽, 高)；不是 PNG / JPEG 或文件损坏时返回 None
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(26)
            if head.startswith(b"\x89PNG"):
                w, h = struct.unpack(">II", head[16:24])
                return w, h
            if head[:2] != b"\xff\xd8":
                return None
            # JPEG：逐段跳过，直到遇到 SOF 段
            fh.seek(2)
            while True:
                marker = fh.read(2)
                if len(marker) < 2 or marker[0] != 0xFF:
                    return None
                if marker[1] in _JPEG_SOF:
                    fh.read(3)
                    h, w = struct.unpack(">HH", fh.read(4))
                    return w, h
                seg_len = struct.unpack(">H", fh.read(2))[0]
                fh.seek(seg_len - 2, 1)
    except (OSError, struct.error):
        return None
