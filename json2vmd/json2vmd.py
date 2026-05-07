"""
JSON -> VMD converter
Usage: python json2vmd.py input.json output.vmd [--flipz]

--flipz: convert right-handed coords (JSON) to VMD left-handed
         position: (x, y, z) -> (x, y, -z)
         quaternion: (qx, qy, qz, qw) -> (-qx, -qy, qz, qw)
"""
import json
import struct
import argparse
from pathlib import Path

SJIS = "shift_jis_2004"


def fixed_str(s: str, length: int) -> bytes:
    encoded = s.encode(SJIS, errors="replace")
    return encoded[:length].ljust(length, b"\x00")


def write_vmd(json_path: str, vmd_path: str, flip_z: bool = False):
    print(f"Reading {json_path} ...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    motions = data["motions"]
    morphs = data.get("morphs", [])

    with open(vmd_path, "wb") as f:
        # --- Header (50 bytes) ---
        f.write(b"Vocaloid Motion Data 0002\x00\x00\x00\x00\x00")  # 30 bytes
        f.write(fixed_str(meta["name"], 20))

        # --- Bone frames ---
        print(f"Writing {len(motions)} bone frames...")
        f.write(struct.pack("<I", len(motions)))
        for m in motions:
            f.write(fixed_str(m["boneName"], 15))
            f.write(struct.pack("<I", m["frameNum"]))

            px, py, pz = m["position"]
            qx, qy, qz, qw = m["rotation"]

            if flip_z:
                pz = -pz
                qx, qy = -qx, -qy

            f.write(struct.pack("<3f", px, py, pz))
            f.write(struct.pack("<4f", qx, qy, qz, qw))

            interp = bytes([v & 0xFF for v in m["interpolation"]])
            interp = interp[:64].ljust(64, b"\x00")
            f.write(interp)

        # --- Morph frames ---
        print(f"Writing {len(morphs)} morph frames...")
        f.write(struct.pack("<I", len(morphs)))
        for m in morphs:
            f.write(fixed_str(m["morphName"], 15))
            f.write(struct.pack("<I", m["frameNum"]))
            f.write(struct.pack("<f", m["weight"]))

        # --- Camera / Light / SelfShadow / ShowIK ---
        f.write(struct.pack("<I", meta.get("cameraCount", 0)))
        f.write(struct.pack("<I", meta.get("lightCount", 0)))
        f.write(struct.pack("<I", meta.get("selfShadowCount", 0)))
        f.write(struct.pack("<I", meta.get("switchFrameCount", 0)))

    size = Path(vmd_path).stat().st_size
    print(f"Done! -> {vmd_path}")
    print(f"File size: {size:,} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert parsed VMD JSON back to .vmd binary")
    parser.add_argument("input",  help="input .json file")
    parser.add_argument("output", help="output .vmd file")
    parser.add_argument("--flipz", action="store_true",
                        help="flip Z axis (right-handed JSON -> left-handed VMD)")
    args = parser.parse_args()
    write_vmd(args.input, args.output, flip_z=args.flipz)
