"""Compatibility support for historical CuRIOS 32-bit JPEG-LS tiles."""

import numpy as np


def _jpeg_ls_start(data, offset):
    """Return whether *offset* begins with a JPEG/JPEG-LS SOI marker."""
    return 0 <= offset <= len(data) - 2 and data[offset : offset + 2] == b"\xff\xd8"


def _split_container_layout(data):
    """Identify a validated current or historical 32-bit tile container."""
    data = bytes(data)
    candidates = []

    if len(data) >= 8:
        upper_len = int.from_bytes(data[:4], "big")
        upper_start = 8
        lower_start = upper_start + upper_len
        upper_valid = upper_len == 0 or _jpeg_ls_start(data, upper_start)
        if upper_len <= len(data) - 8 and upper_valid and _jpeg_ls_start(data, lower_start):
            candidates.append(("current", upper_start, upper_len, int.from_bytes(data[4:8], "big")))

    if len(data) >= 4:
        upper_len = int.from_bytes(data[:4], "big")
        upper_start = 4
        lower_start = upper_start + upper_len
        if upper_len > 0 and upper_len <= len(data) - 4:
            if _jpeg_ls_start(data, upper_start) and _jpeg_ls_start(data, lower_start):
                candidates.append(("legacy-4-byte", upper_start, upper_len, 0))

    if len(data) >= 2:
        upper_len = int.from_bytes(data[:2], "big")
        upper_start = 2
        lower_start = upper_start + upper_len
        if upper_len > 0 and upper_len <= len(data) - 2:
            if _jpeg_ls_start(data, upper_start) and _jpeg_ls_start(data, lower_start):
                candidates.append(("legacy-2-byte", upper_start, upper_len, 0))

    if len(candidates) != 1:
        layouts = ", ".join(item[0] for item in candidates) or "none"
        raise ValueError(
            "Invalid or ambiguous CuRIOS 32-bit JPEG-LS tile container "
            f"(matching layouts: {layouts})"
        )

    return candidates[0]


def install_jpegls_compatibility():
    """Register the backwards-compatible JPEG-LS decoder with Astropy."""
    try:
        from imagecodecs import jpegls_decode
        from astropy.io.fits.hdu.compressed import _tiled_compression
        from astropy.io.fits.hdu.compressed._codecs import JPEGLS as AstropyJPEGLS
    except ImportError:
        return False

    if getattr(_tiled_compression.ALGORITHMS.get("JPEGLS"), "_curios_compat", False):
        return True

    class CompatibleJPEGLS(AstropyJPEGLS):
        _curios_compat = True

        def decode(self, buf):
            if self.bitpix not in (32, -32, -64):
                return super().decode(buf)

            cbytes = np.frombuffer(buf, dtype=np.uint8).tobytes()
            _layout, upper_start, upper_len, baseline = _split_container_layout(cbytes)
            lower_start = upper_start + upper_len
            lower_stream = cbytes[lower_start:]
            split = jpegls_decode(lower_stream).astype(np.uint32).ravel()

            if upper_len > 0:
                upper_stream = cbytes[upper_start:lower_start]
                upper = jpegls_decode(upper_stream).astype(np.uint32).ravel()
                if upper.size != split.size:
                    raise ValueError("JPEG-LS upper and lower planes have different sizes")
                split |= upper << 16

            if self._stream_near(lower_stream) > 0:
                split = np.minimum(split, np.uint32(0xFFFFFFFF) - np.uint32(baseline))

            uval = split + np.uint32(baseline)
            return (uval.astype(np.int64) - 0x80000000).astype(np.int32)

    _tiled_compression.ALGORITHMS["JPEGLS"] = CompatibleJPEGLS
    return True
