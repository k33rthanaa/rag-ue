def iter_rows(files, max_files: int):
    """Yield (title, text) per line across up to max_files files (0 = all).
       Be tolerant of encoding issues by ignoring undecodable bytes."""
    count = 0
    for fp in files:
        if max_files > 0 and count >= max_files:
            break

        opener = gzip.open if fp.endswith(".gz") else open
        mode = "rt" if fp.endswith(".gz") else "r"

        # NOTE: errors="ignore" avoids UnicodeDecodeError on weird bytes
        with opener(fp, mode, encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text  = (obj.get("text")  or "").strip()
                title = (obj.get("title") or "").strip()
                if not text:
                    continue
                yield title, text

        count += 1
