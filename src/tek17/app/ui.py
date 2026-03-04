"""
Streamlit chat client for the TEK17 RAG server.

Run with:
    streamlit run src/tek17/app/ui.py
Or via the CLI:
    python -m tek17 ui
"""
import streamlit as st
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAG_SERVER_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TEK17 RAG",
    page_icon="🏗️",
    layout="wide",
)

st.title("🏗️ TEK17 – Spør om byggeforskrifter")
st.caption("RAG-basert søk i TEK17 (Byggteknisk forskrift) med veiledning fra DiBK.")

# ---------------------------------------------------------------------------
# Sidebar – settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Innstillinger")

    # Check server health
    server_ok = False
    try:
        health = requests.get(f"{RAG_SERVER_URL}/health", timeout=5)
        if health.status_code == 200:
            server_ok = True
            st.success("RAG-server tilkoblet")
        else:
            st.error("RAG-server svarer ikke riktig")
    except requests.ConnectionError:
        st.error("Kan ikke koble til RAG-server. Kjører den på localhost:8000?")

    # Model selection
    available_models = []
    if server_ok:
        try:
            resp = requests.get(f"{RAG_SERVER_URL}/models", timeout=10)
            if resp.status_code == 200:
                available_models = resp.json().get("models", [])
        except Exception:
            pass

    if available_models:
        model = st.selectbox("LLM-modell", available_models, index=0)
    else:
        model = st.text_input("LLM-modell", value="llama3.2")

    top_k = st.slider("Antall kontekst-chunks (top_k)", 1, 20, 6)
    temperature = st.slider("Temperatur", 0.0, 2.0, 0.3, step=0.1)

    # Collection stats
    if server_ok:
        try:
            stats = requests.get(f"{RAG_SERVER_URL}/collection/stats", timeout=5)
            if stats.status_code == 200:
                count = stats.json().get("count", "?")
                st.info(f"Vektordatabase: **{count}** chunks indeksert")
        except Exception:
            pass

    st.divider()
    st.markdown(
        "**Slik bruker du dette:**\n"
        "1. Still et spørsmål om TEK17\n"
        "2. Systemet henter relevante paragrafer\n"
        "3. Ollama genererer et svar basert på konteksten"
    )

# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Kilder"):
                for src in msg["sources"]:
                    section = src.get("section_id", "")
                    title = src.get("title", "")
                    text_type = src.get("text_type", "")
                    chapter = src.get("chapter", "")
                    dist = src.get("distance", 0)

                    label = f"**{section}** – {title}"
                    if text_type:
                        label += f" _({text_type})_"
                    if chapter:
                        label += f" | {chapter}"

                    st.markdown(f"{label}  \nAvstand: `{dist:.4f}`")
                    st.code(src.get("text", "")[:500], language=None)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if question := st.chat_input("Still et spørsmål om TEK17 …"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Query RAG server
    with st.chat_message("assistant"):
        if not server_ok:
            st.error("RAG-serveren er ikke tilgjengelig.")
            st.session_state.messages.append(
                {"role": "assistant", "content": "❌ RAG-serveren er ikke tilgjengelig."}
            )
        else:
            with st.spinner("Søker i TEK17 og genererer svar …"):
                try:
                    resp = requests.post(
                        f"{RAG_SERVER_URL}/query",
                        json={
                            "question": question,
                            "top_k": top_k,
                            "model": model,
                            "temperature": temperature,
                        },
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    answer = data.get("answer", "Ingen svar.")
                    sources = data.get("sources", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander("📚 Kilder"):
                            for src in sources:
                                section = src.get("section_id", "")
                                title = src.get("title", "")
                                text_type = src.get("text_type", "")
                                chapter = src.get("chapter", "")
                                dist = src.get("distance", 0)

                                label = f"**{section}** – {title}"
                                if text_type:
                                    label += f" _({text_type})_"
                                if chapter:
                                    label += f" | {chapter}"

                                st.markdown(f"{label}  \nAvstand: `{dist:.4f}`")
                                st.code(src.get("text", "")[:500], language=None)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )

                except requests.exceptions.HTTPError as e:
                    error_msg = f"Serverfeil: {e}"
                    try:
                        detail = e.response.json().get("detail", "")
                        if detail:
                            error_msg = f"Serverfeil: {detail}"
                    except Exception:
                        pass
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"❌ {error_msg}"}
                    )
                except requests.exceptions.ConnectionError:
                    st.error("Mistet tilkoblingen til RAG-serveren.")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "❌ Mistet tilkoblingen til RAG-serveren."}
                    )
                except Exception as e:
                    st.error(f"Uventet feil: {e}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"❌ Uventet feil: {e}"}
                    )
