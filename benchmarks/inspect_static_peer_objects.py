"""Extract immutable cached CUDA ELFs and census their resource declarations."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess


def cuda_elf(data):
    start = data.find(b"\x7fELF", 1)
    found = []
    while start >= 0:
        if data[start + 4 : start + 6] == b"\x02\x01":
            header = struct.unpack_from("<16sHHIQQQIHHHHHH", data, start)
            if header[2] == 190:
                phoff, shoff = header[5:7]
                ehsize, phentsize, phnum, shentsize, shnum = header[8:13]
                end = max(ehsize, phoff + phentsize * phnum, shoff + shentsize * shnum)
                for i in range(shnum):
                    section = struct.unpack_from(
                        "<IIQQQQIIQQ", data, start + shoff + i * shentsize
                    )
                    if section[1] != 8:  # NOBITS occupies no bytes in the file.
                        end = max(end, section[4] + section[5])
                assert 0 < end <= len(data) - start
                found.append(data[start : start + end])
        start = data.find(b"\x7fELF", start + 4)
    assert len(found) == 1, f"Expected one CUDA ELF, found {len(found)}"
    return found[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for manifest in args.cache.glob("*/*.json"):
        metadata = json.loads(manifest.read_text())
        spec = json.loads(metadata["compile_spec_json"])
        if not spec["kernel"].startswith("comm.pcie.twoshot_bf16.all_reduce_"):
            continue
        obj = manifest.with_suffix(".o")
        data = obj.read_bytes()
        original_hash = hashlib.sha256(data).hexdigest()
        assert original_hash == metadata["object_sha256"]
        blob = cuda_elf(data)
        path = args.out / (obj.stem + ".cubin")
        path.write_bytes(blob)
        raw = subprocess.check_output(
            ["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(path)],
            text=True,
        )
        path.with_suffix(".resources.txt").write_text(raw)
        usage = {
            key: int(value)
            for key, value in re.findall(r"(REG|STACK|SHARED|LOCAL):\s*(\d+)", raw)
        }
        assert {"REG", "STACK", "SHARED", "LOCAL"} <= usage.keys(), raw
        assert hashlib.sha256(obj.read_bytes()).hexdigest() == original_hash
        records.append(
            {
                "manifest": str(manifest),
                "compile_spec": spec,
                "object_sha256": original_hash,
                "cubin_sha256": hashlib.sha256(blob).hexdigest(),
                "usage": usage,
            }
        )
    assert records
    (args.out / "resources.json").write_text(json.dumps(records, indent=2))
    summary = {}
    for record in records:
        key = record["compile_spec"]["kernel"]
        group = summary.setdefault(
            key,
            {
                "objects": 0,
                "min_registers": 10000,
                "max_registers": 0,
                "max_stack": 0,
                "max_local": 0,
            },
        )
        group["objects"] += 1
        group["min_registers"] = min(group["min_registers"], record["usage"]["REG"])
        group["max_registers"] = max(group["max_registers"], record["usage"]["REG"])
        group["max_stack"] = max(group["max_stack"], record["usage"]["STACK"])
        group["max_local"] = max(group["max_local"], record["usage"]["LOCAL"])
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
