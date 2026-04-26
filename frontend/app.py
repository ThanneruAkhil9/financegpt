"""
app.py — Streamlit frontend for FinanceGPT
Deploy to HuggingFace Spaces (SDK: Streamlit)
"""

import streamlit as st
import requests
import uuid
import os

# ─── Config ──────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ─── Page Setup ──────────────────────────────────────────────
st.set_page_config(
    page_title="FinanceGPT 🇮🇳",
    page_icon="💰",
    layout="centered",
)

# ─── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a3a5c, #2e86ab);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        text-align: center;
    }
    .disclaimer-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.8em;
        margin-top: 10px;
    }
    .source-tag {
        background: #e8f4f8;
        color: #1a3a5c;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75em;
        display: inline-block;
        margin: 2px;
    }
    .stButton > button {
        background: #2e86ab;
        color: white;
        border: none;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>💰 FinanceGPT</h1>
    <p>India's Multilingual Finance Assistant | हिंदी • Telugu • English • தமிழ்</p>
</div>
""", unsafe_allow_html=True)

# ─── Session State Init ───────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_sources" not in st.session_state:
    st.session_state.show_sources = False

# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    st.session_state.show_sources = st.toggle(
        "Show source documents", value=st.session_state.show_sources
    )

    st.markdown("---")
    st.markdown("### 💡 Try Asking")
    
    example_questions = [
        "SIP ante enti? (Telugu)",
        "SIP kya hota hai? (Hindi)",
        "What is ELSS and how does 80C work?",
        "Explain LTCG tax on mutual funds",
        "How do I calculate my income tax?",
        "What is Nifty 50 and should I invest?",
        "Difference between direct and regular plans",
        "What is CIBIL score and how to improve it?",
        "How does crypto tax work in India?",
        "What is Mudra Loan?",
    ]
    
    for q in example_questions:
        if st.button(q, key=f"ex_{q[:20]}", use_container_width=True):
            st.session_state.pending_question = q

    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        # Clear memory on backend
        try:
            requests.post(f"{BACKEND_URL}/clear", json={
                "session_id": st.session_state.session_id
            }, timeout=5)
        except Exception:
            pass
        # Reset frontend state
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.markdown("---")
    st.markdown("**Session ID:**")
    st.code(st.session_state.session_id[:8] + "...", language=None)

    st.markdown("""
    <div class='disclaimer-box'>
    ⚠️ For educational purposes only. Not investment advice. 
    Always consult a SEBI-registered adviser.
    </div>
    """, unsafe_allow_html=True)

# ─── Display Chat History ─────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Show sources if enabled and available
        if msg["role"] == "assistant" and st.session_state.show_sources and msg.get("sources"):
            with st.expander("📚 Sources used"):
                for source in msg["sources"]:
                    domain = source.get("domain", "").replace("_", " ").title()
                    title  = source.get("title", "")
                    st.markdown(f"<span class='source-tag'>📁 {domain}</span> {title}", 
                                unsafe_allow_html=True)

# ─── Welcome Message ──────────────────────────────────────────
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("""
Namaste! 🙏 I'm **FinanceGPT** — your India-first finance assistant.

I can help you with:
- 📈 Mutual Funds, SIPs, NAV, ELSS
- 💹 Stock Market, Nifty, IPOs
- 🧾 Income Tax, 80C deductions, ITR
- 🏦 FDs, Banking, RBI, Bonds
- 🏠 Home Loans, EMI, CIBIL Score
- 📊 Personal Finance, Budgeting, Retirement
- 🪙 Crypto Tax in India

**Ask me in any language** — Hindi, Telugu, Tamil, or English!

Try: *"SIP ante enti?"* or *"What is ELSS?"*
        """)

# ─── Handle Example Question Click ────────────────────────────
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    
    with st.chat_message("user"):
        st.markdown(question)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"session_id": st.session_state.session_id, "message": question},
                    timeout=30,
                )
                data = resp.json()
                answer  = data.get("answer", "Sorry, something went wrong.")
                sources = data.get("sources", [])
                st.session_state.session_id = data.get("session_id", st.session_state.session_id)
            except requests.exceptions.ConnectionError:
                answer  = "⚠️ Cannot connect to backend. Make sure the API server is running."
                sources = []
            except Exception as e:
                answer  = f"⚠️ Error: {str(e)}"
                sources = []

        st.markdown(answer)
        if st.session_state.show_sources and sources:
            with st.expander("📚 Sources used"):
                for source in sources:
                    domain = source.get("domain", "").replace("_", " ").title()
                    title  = source.get("title", "")
                    st.markdown(f"<span class='source-tag'>📁 {domain}</span> {title}",
                                unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "sources": sources
    })
    st.rerun()

# ─── Chat Input ───────────────────────────────────────────────
if prompt := st.chat_input("Ask anything about Indian finance... (any language)"):
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"session_id": st.session_state.session_id, "message": prompt},
                    timeout=30,
                )
                data    = resp.json()
                answer  = data.get("answer", "Sorry, something went wrong.")
                sources = data.get("sources", [])
                st.session_state.session_id = data.get("session_id", st.session_state.session_id)
            except requests.exceptions.ConnectionError:
                answer  = "⚠️ Cannot connect to the backend server. Is it running at: `" + BACKEND_URL + "`?"
                sources = []
            except Exception as e:
                answer  = f"⚠️ Error: {str(e)}"
                sources = []

        st.markdown(answer)
        if st.session_state.show_sources and sources:
            with st.expander("📚 Sources used"):
                for source in sources:
                    domain = source.get("domain", "").replace("_", " ").title()
                    title  = source.get("title", "")
                    st.markdown(f"<span class='source-tag'>📁 {domain}</span> {title}",
                                unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "sources": sources
    })
