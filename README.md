# Inbox Triage Assistant

A simple CLI inbox assistant built with LangChain and LangGraph while I was learning agentic workflows.

This project was my practice after completing the LangChain fundamentals course. The goal was to move beyond just calling an LLM and actually build something that uses tools, memory, and multi-step reasoning more like a real agent.

## What It Does

When you run the script, it automatically reads messages from a local text file at [`messages/messages.txt`](/home/diwas/code/lang_chain/lang101/messages/messages.txt) and classifies them into three priorities:

- `URGENT` - needs action today
- `IMPORTANT` - needs action soon
- `ROUTINE` - everything else

It then shows you the triaged inbox and asks if you want it to draft replies. If you say yes, it creates short professional replies with a tone that matches each priority. Finally, it asks for your approval before saving the replies to [`messages/replies.txt`](/home/diwas/code/lang_chain/lang101/messages/replies.txt).

It is a small but practical terminal tool that feels like a focused personal assistant with proper guardrails because it always asks before doing anything permanent.

## What I Learned

This project helped me practice several key concepts:

- Writing a detailed system prompt to guide multi-step behavior
- Creating and using custom tools for file operations with `read_inbox` and `save_replies`
- Using `InMemorySaver` plus a fixed `thread_id` to maintain conversation memory across turns
- Building a human-in-the-loop approval flow before making changes
- Structuring an agent with clear boundaries between tools, prompt, and memory
- Managing a predictable workflow inside an LLM agent

Even though the project is small, it forced me to think about how to make an agent reliable and safe, especially when it has the ability to write files.

## Project Structure

```text
.
├── agent2.py
└── messages/
    ├── messages.txt
    └── replies.txt
```

## How It Works

The implementation in [`agent2.py`](/home/diwas/code/lang_chain/lang101/agent2.py) follows a simple flow:

1. On startup, the script automatically asks the agent to check the inbox.
2. The agent calls `read_inbox()` to load messages from disk.
3. It classifies each message as `URGENT`, `IMPORTANT`, or `ROUTINE`.
4. It shows the grouped inbox and asks whether you want draft replies.
5. If you approve, it generates replies for each message.
6. It asks for final approval before calling `save_replies()`.
7. Approved replies are written to `messages/replies.txt`.

The agent uses LangGraph's in-memory checkpointing so the conversation can keep context while the script is running.

## How to Run

Make sure you have an Anthropic API key available:

```bash
export ANTHROPIC_API_KEY=sk-...
```

Install the dependencies:

```bash
pip install langchain langgraph langchain-anthropic
```

Run the assistant:

```bash
python agent2.py
```

On startup, it will automatically check your inbox and show the triage. Type `exit` or `quit` to stop.

## Tech Stack

- Python 3
- LangChain
- LangGraph
- Claude Sonnet via `init_chat_model`
- `InMemorySaver` for conversation memory

## Current Limitations

To be honest, there are still a few rough edges:

- The workflow is mostly controlled by a long system prompt instead of proper structured output or a real LangGraph state graph
- Classification still relies heavily on keywords in the prompt
- The script defines dataclasses for structure, but it is not yet enforcing them with `.with_structured_output()`
- Memory only lasts while the script is running

These are exactly the areas I want to improve next.

## Ideas for Future Improvements

- Switch to proper structured output with Pydantic for more reliable triage and drafting
- Convert this into a real LangGraph workflow with explicit nodes and state
- Add support for loading emails from Gmail or Outlook
- Add per-message approval instead of batch approval
- Persist memory so the conversation survives script restarts

## Final Note

This was a fun mini-project to solidify what I learned in the LangChain course. It is not perfect, but it is a solid first step into building agentic applications that actually do something useful instead of just chatting.

Feel free to use it as inspiration for your own practice projects.
