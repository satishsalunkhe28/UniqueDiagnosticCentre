import streamlit as st
import chromadb
import uuid
import datetime
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

# ============================================================
# 0. PAGE CONFIG + STYLING  (this is the "beauty" layer)
# ============================================================
st.set_page_config(
    page_title="Unique Diagnostic Centre — AI Assistant",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* overall app background */
.stApp {
    background: linear-gradient(180deg, #f4faff 0%, #ffffff 100%);
}
/* header */
.clinic-header {
    background: linear-gradient(90deg, #0f766e, #14b8a6);
    padding: 22px 28px;
    border-radius: 16px;
    margin-bottom: 18px;
    box-shadow: 0 4px 14px rgba(15,118,110,0.25);
}
.clinic-header h1 {
    color: white;
    font-size: 26px;
    margin: 0;
}
.clinic-header p {
    color: #d1fae5;
    margin: 4px 0 0 0;
    font-size: 14px;
}
/* chat bubbles */
[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 4px 6px;
}
/* sidebar */
[data-testid="stSidebar"] {
    background: #0f172a;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
/* quick action buttons */
div.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: 1px solid #14b8a6;
    background-color: white;
    color: #0f766e;
    font-weight: 600;
    padding: 8px 4px;
}
div.stButton > button:hover {
    background-color: #14b8a6;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="clinic-header">
    <h1>🩺 Unique Diagnostic Centre — AI Assistant</h1>
    <p>Ask about tests, health packages, facilities, or book an appointment — 24/7.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 1. VECTOR DB SETUP (Chroma) — stores bookings + is searchable
# ============================================================
# Chroma's default embedding function (onnx MiniLM) is bundled in,
# so no extra API key or model download is needed for embeddings.
chroma_client = chromadb.PersistentClient(path="./clinic_vector_db")
bookings_collection = chroma_client.get_or_create_collection(name="bookings")

# NOTE: On Google Colab the filesystem is wiped when the runtime
# disconnects, so "./clinic_vector_db" will NOT survive across
# sessions. For a real deployment, point `path=` at a mounted
# Google Drive folder, or swap PersistentClient for a hosted
# vector DB (Chroma Cloud, Pinecone, Qdrant, Weaviate, etc.) —
# same collection.add()/query() calls, just a different client.

# ============================================================
# 2. CLINIC KNOWLEDGE BASE (small + static -> lives in the prompt)
# ============================================================
CLINIC_KNOWLEDGE = """
CLINIC NAME: Unique Diagnostic Centre
ESTABLISHED: 2012
LOCATION: Shop No - 5, Plot No - 2, Maruti Park CHS, Sector-12A, Koparkhairane, Navi Mumbai - 400709
PHONE: +91 97024 72247
EMAIL: info@uniquediagnosticcentre.com

ABOUT:
Unique Diagnostic Centre is a premier pathology service provider based in Navi Mumbai,
offering high-quality, affordable diagnostic services with over 5 lakh happy clients.
Known for precise results, transparency, and doorstep sample collection.

SERVICES (individual tests):
- Haemoglobin Test
- Diabetes Test
- Hormones Profile
- Lipid Profile
- Thyroid Profile
- Vitamins Test
- Kidney Test
- Liver Test

NEW SERVICES:
- Annual Health Checkup
- Pre-Employment Medical Checkup
- Pre-Insurance Medical Checkup
- Occupational Health Services

HEALTH PACKAGES:
- Female Health Package: CBC, Lipid Profile, Liver & Kidney Function, Hormones, Vitamins
- Male Health Package: CBC, Lipid Profile, Testosterone, Kidney & Liver Function Tests
- Diabetic Profile: HbA1c, Lipid Profile, diabetes monitoring & management
- Arthritis Profile: Anti-CCP, RA Factor, Vitamins, and more
- Cardiac Injury Profile: TROP I, TROP T, heart health markers
- Unique Health Profile: CBC, liver, kidney, lipids, essential vitamins (extensive checkup)

FACILITIES:
- X-Ray
- ECG
- TMT (Stress Test)
- PMT (Pulmonary Function Test)

WHAT MAKES US DIFFERENT:
- Convenient home sample collection
- Affordable and comprehensive health packages
- Advanced technology and accurate results
- 24/7 customer support
- Exclusive membership benefits
- Coverage across Navi Mumbai
"""

# ============================================================
# 3. TOOLS
# ============================================================
@tool
def book_appointment(patient_name: str, phone_number: str, test_or_package: str,
                      preferred_date: str = "not specified") -> str:
    """Use this tool whenever the user wants to book an appointment or schedule a
    test. Call it once you know at least their name, phone number, and which
    test/package they want. Ask the user for any missing detail before calling
    this tool. This SAVES the booking into the clinic's database."""

    booking_id = str(uuid.uuid4())[:8]
    timestamp = datetime.datetime.now().isoformat()

    # store as a natural-language document so it is semantically searchable later,
    # plus structured metadata for exact filtering/lookup
    document_text = (
        f"Booking for {patient_name}, phone {phone_number}, "
        f"requested {test_or_package}, preferred date {preferred_date}."
    )
    bookings_collection.add(
        ids=[booking_id],
        documents=[document_text],
        metadatas=[{
            "patient_name": patient_name,
            "phone_number": phone_number,
            "test_or_package": test_or_package,
            "preferred_date": preferred_date,
            "created_at": timestamp,
            "status": "pending_confirmation",
        }],
    )

    return (
        f"✅ Booking recorded! (Reference ID: {booking_id})\n\n"
        f"👤 Name: {patient_name}\n"
        f"📞 Phone: {phone_number}\n"
        f"🧪 Test/Package: {test_or_package}\n"
        f"📅 Preferred date: {preferred_date}\n\n"
        f"Our team will call you on {phone_number} to confirm the exact slot. "
        f"For home sample collection, mention it when they call, or contact us "
        f"directly at +91 97024 72247."
    )


@tool
def lookup_booking(phone_number: str) -> str:
    """Use this tool when a returning user asks about their existing booking
    status, and gives you their phone number."""
    results = bookings_collection.get(where={"phone_number": phone_number})
    if not results["ids"]:
        return f"No booking found for phone number {phone_number}. Would you like to make one?"

    lines = ["Here's what I found:"]
    for meta in results["metadatas"]:
        lines.append(
            f"- {meta['test_or_package']} for {meta['patient_name']} "
            f"(preferred date: {meta['preferred_date']}, status: {meta['status']})"
        )
    return "\n".join(lines)


tools = [book_appointment, lookup_booking]

# ============================================================
# 4. SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Settings")
    user_api_key = st.text_input("Groq API Key:", type="password")
    st.caption("Get a free key at console.groq.com")

    st.divider()
    st.header("📋 Quick Info")
    st.markdown("**📍 Navi Mumbai**  \n**📞 +91 97024 72247**  \n**🕐 24/7 Support**")

    st.divider()
    st.header("🔎 Admin: Recent Bookings")
    if st.button("Refresh bookings"):
        all_bookings = bookings_collection.get()
        if all_bookings["ids"]:
            for meta in all_bookings["metadatas"]:
                st.write(f"• {meta['patient_name']} — {meta['test_or_package']} ({meta['status']})")
        else:
            st.write("No bookings yet.")

# ============================================================
# 5. CHAT UI
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# quick-action buttons above the chat, for a nicer first impression
if not st.session_state.messages:
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🧪 View tests"):
            st.session_state.pending_query = "What tests do you offer?"
    with col2:
        if st.button("📦 Health packages"):
            st.session_state.pending_query = "Tell me about your health packages"
    with col3:
        if st.button("📅 Book appointment"):
            st.session_state.pending_query = "I want to book an appointment"

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🩺"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

typed_query = st.chat_input("Ask about our tests, packages, or book an appointment...")
pending_query = typed_query or st.session_state.get("pending_query")
st.session_state.pending_query = None  # consume it so it only fires once

if pending_query:
    if not user_api_key:
        st.error("Please enter your Groq API Key in the sidebar")
    else:
        st.session_state.messages.append({"role": "user", "content": pending_query})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(pending_query)

        llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=user_api_key)

        system_prompt_text = f"""
        You are the official AI assistant for Unique Diagnostic Centre.

        STRICT RULE: Only answer using the clinic information provided below,
        or by using the book_appointment / lookup_booking tools when relevant.
        Do NOT answer general medical/health questions unrelated to this clinic.
        If asked something outside this clinic's information, politely say you
        can only help with questions about Unique Diagnostic Centre, and suggest
        they call +91 97024 72247 for anything else.

        When a user wants to book, collect their name, phone number, and the
        test/package they want before calling book_appointment.

        CLINIC INFORMATION:
        {CLINIC_KNOWLEDGE}
        """
        system_message = SystemMessage(content=system_prompt_text)
        agent = create_react_agent(llm, tools, prompt=system_message)

        langgraph_history = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                langgraph_history.append(HumanMessage(content=msg["content"]))
            else:
                langgraph_history.append(AIMessage(content=msg["content"]))

        with st.chat_message("assistant", avatar="🩺"):
            with st.spinner("🤖 Checking clinic details..."):
                result_state = agent.invoke({"messages": langgraph_history})
                bot_answer = result_state["messages"][-1].content
            st.markdown(bot_answer)

        st.session_state.messages.append({"role": "assistant", "content": bot_answer})
