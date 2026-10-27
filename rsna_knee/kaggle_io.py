"""Robust download of Kaggle kernel outputs: streaming, HTTP Range resume, retries, fresh signed URLs.

The stock CLI (`kaggle kernels output`) reads each file fully into RAM and restarts from zero on any hiccup,
which fails on multi-GB cache tars over a flaky link.
"""
from __future__ import annotations

import os
import re
import time
from typing import Callable

import requests


def list_kernel_files(kernel: str, pattern: str | None = None) -> dict[str, str]:
    """{file_name: signed_url} for the latest run of ``owner/slug`` (all pages)."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    api = KaggleApi()
    api.authenticate()
    owner, slug = kernel.split("/", 1)
    rx = re.compile(pattern) if pattern else None
    out: dict[str, str] = {}
    token = None
    with api.build_kaggle_client() as client:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name, req.kernel_slug = owner, slug
            req.page_size = 200
            if token:
                req.page_token = token
            resp = client.kernels.kernels_api_client.list_kernel_session_output(req)
            for item in resp.files or []:
                if rx is None or rx.search(item.file_name):
                    out[item.file_name] = item.url
            token = resp.next_page_token
            if not token:
                return out


def download_with_resume(get_url: Callable[[], str], dest: str, tries: int = 12, chunk: int = 1 << 22, log=print) -> str:
    """Stream ``get_url()`` to ``dest``; on failure resume from the bytes already on disk (``dest.part``)."""
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    total = None
    for attempt in range(1, tries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        try:
            url = get_url()  # signed URLs expire: ask for a fresh one on every attempt
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, stream=True, headers=headers, timeout=(30, 180)) as r:
                if r.status_code == 416 and have:  # range not satisfiable: we already hold the whole file
                    total = have
                else:
                    r.raise_for_status()
                    if have and r.status_code != 206:  # server ignored the range: start over
                        have = 0
                    rng = r.headers.get("Content-Range")  # "bytes a-b/total"
                    total = int(rng.rsplit("/", 1)[1]) if rng and "/" in rng else have + int(r.headers.get("Content-Length", 0))
                    t0, last = time.time(), time.time()
                    with open(part, "ab" if have else "wb") as f:
                        for block in r.iter_content(chunk):
                            f.write(block)
                            have += len(block)
                            if time.time() - last > 30:
                                last = time.time()
                                log(f"  {os.path.basename(dest)}: {have / 1e9:.2f}/{total / 1e9:.2f} GB, {(time.time() - t0):.0f}s")
            if total and os.path.getsize(part) >= total:
                os.replace(part, dest)
                return dest
            log(f"  short read ({os.path.getsize(part)}/{total}), resuming (attempt {attempt}/{tries})")
        except (requests.RequestException, OSError) as e:
            log(f"  download error ({type(e).__name__}: {str(e)[:120]}), resuming (attempt {attempt}/{tries})")
        time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"could not download {dest} after {tries} attempts")


def download_kernel_outputs(kernel: str, dest_dir: str, pattern: str | None = None, log=print) -> list[str]:
    files = list_kernel_files(kernel, pattern)
    if not files:
        raise RuntimeError(f"no output files match {pattern!r} in {kernel}")
    done = []
    for name in sorted(files):
        dest = os.path.join(dest_dir, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            log(f"have {name}")
        else:
            log(f"downloading {name} from {kernel}")
            download_with_resume(lambda n=name: list_kernel_files(kernel, pattern)[n], dest, log=log)
        done.append(dest)
    return done
