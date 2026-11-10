#!/usr/bin/env python3
"""Upload one file to the HF storage repo from inside the training container.

Reads the token from env HF_TOKEN and the repo from env HF_REPO
(default Works403/lora-storage). Implements the Git LFS flow used by
huggingface_hub: batch -> PUT blob -> commit (key "lfsFile").

Usage: hf_put.py <local_path> <repo_path> [summary]
"""
import hashlib
import json
import os
import sys
import urllib.request
import urllib.error

TOKEN = os.environ.get("HF_TOKEN", "")
REPO = os.environ.get("HF_REPO", "Works403/lora-storage")
UA = {"User-Agent": "salad-lora-hf-put/1.0"}


def http(method, url, data=None, headers=None, auth=False, timeout=120):
    h = {**UA, **(headers or {})}
    if auth:
        h["Authorization"] = f"Bearer {TOKEN}"
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not TOKEN:
        print("ERROR: HF_TOKEN is not set", flush=True)
        return 2
    local, repo_path = sys.argv[1], sys.argv[2]
    summary = sys.argv[3] if len(sys.argv) > 3 else f"Upload {repo_path}"
    size = os.path.getsize(local)
    oid = sha256_of(local)
    print(f"[hf_put] {local} size={size} sha256={oid}", flush=True)

    batch_body = json.dumps({
        "operation": "upload",
        "transfers": ["basic"],
        "objects": [{"oid": oid, "size": size}],
        "hash_algo": "sha256",
    }).encode()
    status, raw = http(
        "POST",
        f"https://huggingface.co/{REPO}.git/info/lfs/objects/batch",
        data=batch_body,
        headers={"Accept": "application/vnd.git-lfs+json",
                 "Content-Type": "application/vnd.git-lfs+json"},
        auth=True,
    )
    if status >= 300:
        print(f"[hf_put] ERROR lfs batch {status}: {raw[:500]!r}", flush=True)
        return 1
    obj = json.loads(raw)["objects"][0]
    if "error" in obj:
        print(f"[hf_put] ERROR lfs object: {obj['error']}", flush=True)
        return 1
    upload = (obj.get("actions") or {}).get("upload")
    if upload:
        with open(local, "rb") as f:
            status, raw = http("PUT", upload["href"], data=f,
                               headers=upload.get("header") or {}, timeout=1200)
        if status >= 300:
            print(f"[hf_put] ERROR blob PUT {status}: {raw[:500]!r}", flush=True)
            return 1
        print("[hf_put] blob uploaded", flush=True)
    else:
        print("[hf_put] blob already present (dedup)", flush=True)

    lines = [
        {"key": "header", "value": {"summary": summary}},
        {"key": "lfsFile", "value": {"path": repo_path, "oid": oid, "size": size}},
    ]
    ndjson = ("\n".join(json.dumps(x) for x in lines)).encode()
    status, raw = http(
        "POST",
        f"https://huggingface.co/api/models/{REPO}/commit/main",
        data=ndjson,
        headers={"Content-Type": "application/x-ndjson"},
        auth=True,
    )
    print(f"[hf_put] commit status {status}: {raw[:300]!r}", flush=True)
    return 0 if status < 300 else 1


if __name__ == "__main__":
    sys.exit(main())
