# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Generate OpenCV pixel oracles for the ignored NM4 video decoder test.

Requires ffmpeg, ffprobe, numpy, and opencv-python-headless. Pass an output
directory as the only argument, then set NM4_VIDEO_FIXTURES to that directory
when running encoded_video_frames_match_opencv_oracle with --ignored.
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def ffmpeg(*arguments: str) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *arguments],
        check=True,
    )


def generate(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    colors = ((0, 0, 255), (0, 128, 0), (255, 0, 0))
    for index, color in enumerate(colors):
        frame = np.full((256, 384, 3), color, dtype=np.uint8)
        assert cv2.imwrite(str(root / f"frame-{index}.png"), frame)
    ffmpeg(
        "-framerate",
        "1",
        "-i",
        str(root / "frame-%d.png"),
        "-frames:v",
        "3",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "1",
        str(root / "colors.mp4"),
    )

    frame = np.full((256, 384, 3), colors[0], dtype=np.uint8)
    frame[:, 192:] = colors[2]
    assert cv2.imwrite(str(root / "rotation-source.png"), frame)
    ffmpeg(
        "-loop",
        "1",
        "-i",
        str(root / "rotation-source.png"),
        "-t",
        "3",
        "-r",
        "1",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(root / "rotation-source.mp4"),
    )
    ffmpeg(
        "-display_rotation:v:0",
        "90",
        "-i",
        str(root / "rotation-source.mp4"),
        "-c",
        "copy",
        str(root / "rotated.mp4"),
    )
    for index in range(12):
        frame = np.full((256, 384, 3), colors[index // 4], dtype=np.uint8)
        assert cv2.imwrite(str(root / f"trim-frame-{index:02}.png"), frame)
    ffmpeg(
        "-framerate",
        "2",
        "-i",
        str(root / "trim-frame-%02d.png"),
        "-frames:v",
        "12",
        "-c:v",
        "libx264",
        "-g",
        "12",
        "-pix_fmt",
        "yuv420p",
        str(root / "trim-source.mp4"),
    )
    ffmpeg(
        "-ss",
        "1.5",
        "-i",
        str(root / "trim-source.mp4"),
        "-c",
        "copy",
        str(root / "trimmed.mp4"),
    )

    cases = []
    for name in ("colors.mp4", "rotated.mp4", "trimmed.mp4"):
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=width,height,nb_frames,nb_read_frames:stream_side_data=rotation",
                    "-of",
                    "json",
                    str(root / name),
                ],
                text=True,
            )
        )["streams"][0]
        capture = cv2.VideoCapture(str(root / name))
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            valid, frame = capture.read()
            if not valid:
                break
            frames.append(frame)
        capture.release()
        assert frames and source_fps > 0, name
        assert len(frames) == int(probe["nb_read_frames"]), probe
        if name == "rotated.mp4":
            assert frames[0].shape[:2] == (384, 256), probe
        if name == "trimmed.mp4":
            assert int(probe["nb_frames"]) > len(frames), probe
        count = max(1, min(3, len(frames), int(len(frames) / source_fps)))
        selected = []
        for index in np.linspace(0, len(frames) - 1, count, dtype=int).tolist():
            expected = f"{Path(name).stem}-expected-{index}.png"
            assert cv2.imwrite(str(root / expected), frames[index])
            selected.append({"index": index, "file": expected})
        cases.append(
            {
                "file": name,
                "num_frames": 3,
                "fps": 1,
                "source_fps": source_fps,
                "visible_frames": len(frames),
                "metadata": probe,
                "frames": selected,
            }
        )
    (root / "decode-oracle.json").write_text(json.dumps(cases, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    generate(parser.parse_args().output)
