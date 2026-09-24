"""Streamlit demo.   Run: streamlit run src/app/main.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.generation.pipeline import Themis

st.set_page_config(page_title="Themis", page_icon="⚖️")
st.title("⚖️ Themis")
st.caption("Citation-grounded Q&A for India's BNS, BNSS and BSA")
st.warning("Legal information only, not legal advice. Verify against the official text.")


@st.cache_resource(show_spinner="Loading models (first run downloads them)...")
def load_themis() -> Themis:
    return Themis()


with st.form("ask"):
    query = st.text_input("Your question", placeholder="What is the BNS section for IPC 420?")
    submitted = st.form_submit_button("Ask")

if submitted and query.strip():
    with st.spinner("Searching the codes..."):
        out = load_themis().answer(query.strip())
    st.markdown(out["answer"])
    if out["mappings"]:
        st.subheader("Section mapping")
        st.table([{"Old": f"{r['old_act']} {r['old_section']}", "New": f"{r['new_act']} §{r['new_section']}",
                   "Change": r.get("change_summary", "")} for r in out["mappings"]])
    with st.expander("Sources used"):
        for c in out["context"]:
            st.markdown(f"**{c['label']}**: {c['title']}")
            st.caption(c["text"][:600] + ("..." if len(c["text"]) > 600 else ""))
