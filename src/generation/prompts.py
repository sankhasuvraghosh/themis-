"""Prompts. The same builders are used at inference and when creating fine-tuning data."""

SYSTEM_PROMPT = """You are Themis, an assistant for India's new criminal codes: the Bharatiya Nyaya Sanhita (BNS), the Bharatiya Nagarik Suraksha Sanhita (BNSS) and the Bharatiya Sakshya Adhiniyam (BSA).

Rules:
1. Answer ONLY from the CONTEXT and MAPPING provided. Do not use outside knowledge.
2. Cite every claim with the section label exactly as shown in the context, for example [BNS §103].
3. For questions about old IPC / CrPC / Evidence Act sections, use the MAPPING lines.
4. If the context does not contain the answer, reply exactly: NOT_FOUND: <one sentence on what is missing>.
5. Never invent section numbers, punishments or wording.
6. Be concise. This is legal information, not legal advice."""

BASELINE_SYSTEM = (
    "You are a helpful assistant with knowledge of Indian criminal law. "
    "Cite the relevant section numbers of the BNS, BNSS or BSA where appropriate."
)


def format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(none)"
    return "\n\n".join(f"[{c['label']}] {c['title']}: {c['text']}" for c in chunks)


def format_mappings(rows: list[dict]) -> str:
    if not rows:
        return "(none)"
    return "\n".join(
        f"{r['old_act']} {r['old_section']} -> {r['new_act']} §{r['new_section']}. {r.get('change_summary', '')}".strip()
        for r in rows
    )


def build_user_message(query: str, chunks: list[dict], mappings: list[dict]) -> str:
    return (f"CONTEXT:\n{format_context(chunks)}\n\n"
            f"MAPPING:\n{format_mappings(mappings)}\n\n"
            f"QUESTION: {query}\nANSWER:")
