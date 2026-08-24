import numpy as np
import cv2
from PIL import Image
from imwatermark import WatermarkEncoder

# Fixed 32-bit payload identifying EDR outputs.
PAYLOAD = np.array([int(b) for b in format(0xE0D12026, "032b")], dtype=np.uint8)

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        WatermarkEncoder.loadModel()
        _encoder = WatermarkEncoder()
        _encoder.set_watermark("bits", PAYLOAD.tolist())
    return _encoder


def embed(pil_img: Image.Image) -> Image.Image:
    """Invisible blind watermark (Zhang et al., 2019), applied by default to every output."""
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    bgr = _get_encoder().encode(bgr, "rivaGan")
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
