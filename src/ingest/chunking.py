"""Turn section records into retrieval chunks (one chunk per section; long ones are split)."""


def chunk_section(sec: dict, max_chars: int = 1500, overlap: int = 150) -> list[dict]:
    body = sec["text"] or sec["title"]
    parts, start = [], 0
    while start < len(body):
        end = min(start + max_chars, len(body))
        if end < len(body):
            cut = body.rfind(" ", start + max_chars // 2, end)
            if cut != -1:
                end = cut
        parts.append(body[start:end].strip())
        if end >= len(body):
            break
        start = max(end - overlap, start + 1)

    header = f'{sec["act"]} Section {sec["section"]}: {sec["title"]}.'
    chunks = []
    for i, part in enumerate(parts or [""]):
        chunks.append({
            **sec,
            "part": i,
            "id": f'{sec["act"]}-{sec["section"]}' + (f"-p{i}" if len(parts) > 1 else ""),
            "label": f'{sec["act"]} §{sec["section"]}',
            "text": part,
            "embed_text": f"{header} {part}",
        })
    return chunks
