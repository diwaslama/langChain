from dataclasses import dataclass, field
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy

# ── System prompt ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are an inbox triage assistant. No emojis.

When asked to check the inbox, do the following in order:

STEP 1 — Call read_inbox and classify every message into:
  URGENT   — action required today, real consequence if ignored
             (keywords: URGENT, ASAP, emergency, system down, suspended)
  IMPORTANT — action required but not today
             (keywords: Important, reminder, deadline within days)
  ROUTINE  — everything else, default when uncertain

STEP 2 — Present the triaged inbox grouped by priority, then ask:
"Would you like me to draft replies? (yes/no)"
Wait for user response.

STEP 3 — If user says yes, draft a short professional reply for every message.
Reply tone guide:
  urgent    → direct, acknowledge urgency, confirm action
  important → professional, acknowledge, give a timeframe
  routine   → friendly, casual, brief

Present all drafts grouped by priority exactly like this:

--- URGENT ---
2. [original message]
Reply: [your draft]

--- IMPORTANT ---
...

--- ROUTINE ---
...

Then ask: "Approve and send these replies? (yes/no)"
Wait for user response.

STEP 4 — If user says yes, call save_replies with all drafts in order.
         Do not include the word "Reply:" or "Draft:" in the saved text,
         only the plain reply text prefixed by the message number.
         If user says no, tell them the drafts have been discarded."""

# ── Tools ──────────────────────────────────────────────────────
@tool
def read_inbox() -> str:
    """Read all messages from the inbox file."""
    with open("messages/messages.txt", "r") as f:
        return f.read()

@tool
def save_replies(replies: list[str]) -> str:
    """Save approved reply drafts to replies.txt.
    Replies must be ordered: urgent first, then important, then routine.
    Each reply should include the message number and plain reply text only."""
    with open("messages/replies.txt", "w") as f:
        for reply in replies:
            clean = reply.replace("Reply:", "").replace("Draft:", "").strip()
            f.write(clean + "\n\n")
    return "replies.txt saved successfully."

# ── Structured output ──────────────────────────────────────────
@dataclass
class Message:
    number: int
    content: str
    priority: str  # "urgent" | "important" | "routine"
    reason: str
    draft_reply: str

@dataclass
class InboxResponse:
    urgent: list[Message] = field(default_factory=list)
    important: list[Message] = field(default_factory=list)
    routine: list[Message] = field(default_factory=list)
    summary: str = ""

# ── Model ──────────────────────────────────────────────────────
load_dotenv()
model = init_chat_model(
    "gpt-5.4",
    temperature=0
)

# ── Memory ─────────────────────────────────────────────────────
checkpointer = InMemorySaver()

# ── Agent ──────────────────────────────────────────────────────
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[read_inbox, save_replies],
    checkpointer=checkpointer
)

# ── Conversation loop ──────────────────────────────────────────
config = {"configurable": {"thread_id": "inbox-1"}}

print("Inbox Assistant ready.\n")

# trigger triage automatically on first run
initial_response = agent.invoke(
    {"messages": [{"role": "user", "content": "can you check my inbox"}]},
    config=config
)
print(f"Agent: {initial_response['messages'][-1].content}\n")

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye.")
        break

    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config
    )

    print(f"\nAgent: {response['messages'][-1].content}\n")