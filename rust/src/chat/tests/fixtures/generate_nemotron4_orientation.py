# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Generate lossless EXIF-orientation references using Pillow."""

import io
import json
from pathlib import Path

import pybase64 as base64
from PIL import Image, ImageOps


def png(image: Image.Image, **kwargs: object) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", **kwargs)
    return output.getvalue()


image = Image.new("RGB", (56, 28))
image.putdata([(x * 4, y * 8, (x + y) * 3) for y in range(28) for x in range(56)])
fixtures = []
for orientation in range(1, 9):
    exif = Image.Exif()
    exif[274] = orientation
    encoded = png(image, exif=exif)
    expected = ImageOps.exif_transpose(Image.open(io.BytesIO(encoded)))
    fixtures.append(
        {
            "orientation": orientation,
            "input": base64.b64encode(encoded).decode(),
            "expected": base64.b64encode(png(expected)).decode(),
        }
    )
Path(__file__).with_name("nemotron4_orientation.json").write_text(
    json.dumps(fixtures, separators=(",", ":")) + "\n"
)
