import boto3
import streamlit as st
import uuid
from botocore.config import Config

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TravelMind — AI Travel Planner",
    page_icon="✈️",
    layout="centered"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>✈️ TravelMind</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered travel budget planner — tell me where you want to go!</div>', unsafe_allow_html=True)

# ── Bedrock client ────────────────────────────────────────────────────────────
@st.cache_resource
def get_bedrock_client():
    config = Config(
        read_timeout=600,       # 10 minutes — agent calls 7 sub-agents sequentially
        connect_timeout=10,
        retries={"max_attempts": 0},
    )
    return boto3.client(
        service_name="bedrock-agent-runtime",
        region_name=st.secrets["AWS_REGION"],
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        config=config,
    )

AGENT_ID       = st.secrets["BEDROCK_AGENT_ID"]        # Supervisor agent ID
AGENT_ALIAS_ID = st.secrets["BEDROCK_AGENT_ALIAS_ID"]  # v1 alias ID

def call_agent(user_message: str, session_id: str) -> str:
    """Invoke the TravelMind Supervisor agent and return the full response."""
    client = get_bedrock_client()

    response = client.invoke_agent(
        agentId=AGENT_ID,
        agentAliasId=AGENT_ALIAS_ID,
        sessionId=session_id,
        inputText=user_message,
    )

    full_text = ""
    for event in response.get("completion", []):
        if "chunk" in event:
            chunk_bytes = event["chunk"].get("bytes", b"")
            full_text += chunk_bytes.decode("utf-8")

    return full_text or "I'm sorry, I didn't receive a response. Please try again."

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🗺️ How it works")
    st.markdown("""
    1. **Tell me your destination** — city, country, or region
    2. **Share your budget** in USD
    3. **Set your trip duration** — days and nights
    4. **Rank your priorities** — hotel, food, activities, transport
    5. **Get your itinerary** — day-by-day plan in your budget!
    """)

    st.divider()
    st.markdown("**Example prompts:**")
    st.code("Plan a 3-day trip to Ubud, Bali for $300")
    st.code("5 days in Bangkok with $500 budget")
    st.code("Weekend in Singapore, $200, 2 days 1 night")

    st.divider()
    if st.button("🔄 Start New Trip", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧳" if message["role"] == "assistant" else None):
        st.markdown(message["content"])

# ── Welcome message ───────────────────────────────────────────────────────────
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🧳"):
        st.markdown("""
Hi! I'm **TravelMind** — your AI travel budget planner. 🌍

Just tell me:
- **Where** you want to go
- **How much** you have to spend (in USD)
- **How long** your trip is

For example: *"I want to plan a 3-day trip to Bali for $300"*

Let's plan your perfect trip! ✈️
        """)

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Where are you travelling to? 🌏"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and show agent response
    with st.chat_message("assistant", avatar="🧳"):
        with st.spinner("🔍 Planning your trip..."):
            try:
                reply = call_agent(prompt, st.session_state.session_id)
            except Exception as e:
                reply = f"⚠️ Something went wrong: {str(e)}\n\nPlease try again."
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
