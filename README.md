# Inbox Triage Assistant

A compact LangChain + LangGraph CLI agent that reads a local inbox, sorts every message by urgency, drafts replies on request, and saves approved responses back to disk.

This project is centered on [`agent2.py`]. It behaves like a focused personal assistant for message cleanup:

- It automatically checks the inbox when the script starts.
- It classifies messages into `URGENT`, `IMPORTANT`, and `ROUTINE`.
- It asks before drafting replies.
- It asks again before saving anything.
- It writes approved drafts to `messages/replies.txt`.

## Why It’s Cool

This is a small script, but it demonstrates several solid agent patterns in one place:

- A clear system prompt that defines a multi-step workflow.
- Tool use for real file I/O.
- Structured state via dataclasses for inbox responses.
- Conversation memory with LangGraph checkpointing.
- A human approval loop before side effects happen.

It feels less like a toy chatbot and more like a practical terminal assistant with guardrails.

## How It Works

When you run the script, the agent immediately receives:

```text
can you check my inbox
```

From there, the prompt instructs the model to follow this flow:

1. Call `read_inbox()`.
2. Classify every message as `URGENT`, `IMPORTANT`, or `ROUTINE`.
3. Show the grouped inbox and ask whether to draft replies.
4. If approved, draft replies for every message using a tone that matches the priority.
5. Ask for final approval before saving.
6. If approved, call `save_replies()` and write the cleaned replies to disk.

The conversation state is preserved in memory through LangGraph's `InMemorySaver`, using the fixed thread ID `inbox-1`.

## Project Structure

```text
.
├── agent2.py
└── messages
    ├── messages.txt
    └── replies.txt
```

## Core Components

### `SYSTEM_PROMPT`

The heart of the assistant is a strict workflow prompt that tells the model exactly how to:

- triage the inbox
- ask for consent before generating replies
- ask for consent again before saving
- format final output

That makes the agent predictable and easy to reason about.

### `read_inbox()`

Reads the raw inbox from:

[`messages/messages.txt`](/home/diwas/code/lang_chain/lang101/messages/messages.txt)

This is the agent's source of truth for messages.

### `save_replies(replies)`

Writes approved replies to:

[`messages/replies.txt`](/home/diwas/code/lang_chain/lang101/messages/replies.txt)

Before saving, it strips labels like `Reply:` or `Draft:` so the output file only contains clean reply text.

### `Message` and `InboxResponse`

These dataclasses describe the intended response shape:

- each message has a number, content, priority, reason, and draft reply
- the full inbox response groups messages into urgent, important, and routine buckets

They make the script easier to extend toward stricter structured output later.

### Model Setup

The agent is initialized with:

```python
model = init_chat_model("claude-sonnet-4-6", temperature=0)
```

So the current implementation is wired for Claude through LangChain's Anthropic integration, with deterministic behavior favored over creativity.

## Requirements

This repo is currently running with:

- Python `3.14.3`
- `langchain==1.2.13`
- `langgraph==1.1.3`
- `langchain-anthropic==1.4.0`

You will also need an Anthropic API key available to the LangChain Anthropic integration, typically via:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

## Run It

From the project root:

```bash
python agent2.py
```

On startup, you should see:

```text
Inbox Assistant ready.
```

Then the assistant will automatically triage the inbox before waiting for your next input.

Type `exit` or `quit` to leave the session.

## Example Inbox

The sample inbox currently includes messages like:

- `URGENT: The server is down, we need you NOW.`
- `ASAP: Client meeting moved to 9am tomorrow, confirm attendance.`
- `Important: Your payment is overdue, account will be suspended in 24 hours.`
- casual personal check-ins and reminders

That mix is useful because it shows the agent handling both operational alerts and everyday messages in one pass.

## Example Flow

Typical interaction:

```text
Agent: [triaged inbox grouped by priority]
Agent: Would you like me to draft replies? (yes/no)

You: yes

Agent: [draft replies grouped by priority]
Agent: Approve and send these replies? (yes/no)

You: yes

Agent: replies.txt saved successfully.
```

The saved output is ordered exactly as the prompt requires:

1. urgent replies first
2. important replies second
3. routine replies last

## What Makes This README-Worthy

`agent2.py` is a nice example of an opinionated agent loop done right:

- narrow scope
- visible tool boundaries
- human approval before writes
- memory across turns
- simple enough to learn from in one sitting

If you are learning LangChain agents, this is a strong pattern to study because it stays practical without getting bloated.

## Ideas for Next Upgrades

- Load the inbox from email, Slack, or a database instead of a text file.
- Persist memory so the thread survives process restarts.
- Enforce structured output with the existing response dataclasses.
- Add per-message approval instead of batch approval.
- Support custom priority rules or user-defined labels.
- Log sent drafts with timestamps and audit history.

## License

No license file is currently included in this repository.
