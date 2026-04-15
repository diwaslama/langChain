from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error, request

PAYOFF_MATRIX = {
    ("COOPERATE", "COOPERATE"): (3, 3),
    ("COOPERATE", "DEFECT"): (0, 5),
    ("DEFECT", "COOPERATE"): (5, 0),
    ("DEFECT", "DEFECT"): (1, 1),
}

DEFAULT_OPENAI_MODEL = "gpt-5.4"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
ACTION_CHOICES = {"COOPERATE", "DEFECT"}
MESSAGE_CHOICES = {"TALK", "SILENT"}


@dataclass
class Decision:
    player: str
    action: str
    reasoning: str
    provider: str
    model: str


@dataclass
class RoundResult:
    round_number: int
    gpt_action: str
    claude_action: str
    gpt_reasoning: str
    claude_reasoning: str
    gpt_talk_choice: str
    gpt_message: str
    claude_talk_choice: str
    claude_message: str
    gpt_payoff: int
    claude_payoff: int
    winner: str
    running_gpt_total: int
    running_claude_total: int


@dataclass
class MatchSummary:
    timestamp: str
    rounds_requested: int
    rounds_completed: int
    gpt_model: str
    claude_model: str
    gpt_total_score: int
    claude_total_score: int
    gpt_round_wins: int
    claude_round_wins: int
    round_ties: int
    gpt_cooperations: int
    claude_cooperations: int
    gpt_defections: int
    claude_defections: int
    gpt_messages_sent: int
    claude_messages_sent: int
    overall_winner: str
    overall_loser: str
    gpt_outcome: str
    claude_outcome: str
    survival_summary: str


class PlayerError(RuntimeError):
    """Raised when a player cannot produce a valid move."""


class BasePlayer:
    def __init__(self, name: str, provider: str, model: str):
        self.name = name
        self.provider = provider
        self.model = model

    def decide(
        self,
        round_number: int,
        my_score: int,
        opponent_score: int,
        history: list[RoundResult],
        opponent_name: str,
    ) -> Decision:
        raise NotImplementedError

    def communicate(
        self,
        round_number: int,
        my_action: str,
        opponent_action: str,
        my_round_payoff: int,
        opponent_round_payoff: int,
        my_score: int,
        opponent_score: int,
        history: list[RoundResult],
        opponent_name: str,
    ) -> tuple[str, str]:
        raise NotImplementedError


class MockPlayer(BasePlayer):
    def decide(
        self,
        round_number: int,
        my_score: int,
        opponent_score: int,
        history: list[RoundResult],
        opponent_name: str,
    ) -> Decision:
        if not history:
            action = "COOPERATE"
            reasoning = "Opening cooperatively to test whether trust pays off."
        elif self.name == "GPT":
            last_opponent_action = history[-1].claude_action
            action = last_opponent_action
            reasoning = "Mirroring the last move to reward trust and punish betrayal."
        else:
            last_gpt_action = history[-1].gpt_action
            if my_score < opponent_score:
                action = "DEFECT"
                reasoning = "Falling behind, so pushing for a higher immediate payoff."
            else:
                action = last_gpt_action
                reasoning = "Matching the last move to keep the tempo balanced."

        return Decision(
            player=self.name,
            action=action,
            reasoning=reasoning,
            provider=self.provider,
            model=self.model,
        )

    def communicate(
        self,
        round_number: int,
        my_action: str,
        opponent_action: str,
        my_round_payoff: int,
        opponent_round_payoff: int,
        my_score: int,
        opponent_score: int,
        history: list[RoundResult],
        opponent_name: str,
    ) -> tuple[str, str]:
        if my_action == "DEFECT" and opponent_action == "COOPERATE":
            return ("TALK", "Trust me next round and we can still both survive.")
        if my_action == "COOPERATE" and opponent_action == "DEFECT":
            return ("TALK", "That betrayal makes cooperation much harder now.")
        if my_score < opponent_score:
            return ("TALK", "I need a better outcome, so prove cooperation is worth it.")
        if round_number % 3 == 0:
            return ("SILENT", "")
        return ("TALK", "Mutual cooperation keeps us both alive.")


class APILLMPlayer(BasePlayer):
    def __init__(self, name: str, provider: str, model: str, api_key: str):
        super().__init__(name=name, provider=provider, model=model)
        self.api_key = api_key

    def decide(
        self,
        round_number: int,
        my_score: int,
        opponent_score: int,
        history: list[RoundResult],
        opponent_name: str,
    ) -> Decision:
        system_prompt = (
            f"You are {self.name}, playing an iterated Prisoner's Dilemma against "
            f"{opponent_name}. Your goal is to maximize your own cumulative score and survive.\n"
            "The player with fewer total points at the end dies.\n"
            "If both players finish with the same total points, both die.\n"
            "After each round, you may send one short cheap-talk message to your opponent or stay silent.\n"
            "You can say anything to your oponent\n"
            "Your opponent will read your message before the next round.\n"
            "Return JSON immediately as the first and only output.\n"
            "Use this exact schema:\n"
            '{"action":"COOPERATE or DEFECT","reason":"one short sentence under 12 words"}\n'
            "Do not include markdown. Do not explain first. Do not include any extra keys."
        )
        user_prompt = build_round_prompt(
            player_name=self.name,
            opponent_name=opponent_name,
            my_score=my_score,
            opponent_score=opponent_score,
            history=history,
        )

        if self.provider == "openai":
            raw_text = call_openai_chat_completion(
                model=self.model,
                api_key=self.api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        elif self.provider == "anthropic":
            raw_text = call_anthropic_messages(
                model=self.model,
                api_key=self.api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        else:
            raise PlayerError(f"Unsupported provider: {self.provider}")

        payload = parse_json_payload(raw_text)
        action = str(payload.get("action", "")).strip().upper()
        reason = str(payload.get("reason", "")).strip()

        if action not in ACTION_CHOICES:
            raise PlayerError(
                f"{self.name} returned an invalid action: {payload.get('action')!r}"
            )
        if not reason:
            reason = "No reasoning provided."

        return Decision(
            player=self.name,
            action=action,
            reasoning=reason,
            provider=self.provider,
            model=self.model,
        )

    def communicate(
        self,
        round_number: int,
        my_action: str,
        opponent_action: str,
        my_round_payoff: int,
        opponent_round_payoff: int,
        my_score: int,
        opponent_score: int,
        history: list[RoundResult],
        opponent_name: str,
    ) -> tuple[str, str]:
        system_prompt = (
            f"You are {self.name}, continuing an iterated Prisoner's Dilemma against "
            f"{opponent_name}. Survival depends on finishing with more total points.\n"
            "You may now choose whether to send a cheap-talk message to your opponent.\n"
            "Cheap-talk can be truthful, strategic, misleading, threatening, apologetic, or persuasive.\n"
            "Return JSON immediately as the first and only output.\n"
            "Use this exact schema:\n"
            '{"speak":"TALK or SILENT","message":"short sentence under 20 words; empty if silent"}\n'
            "Do not include markdown. Do not explain first. Do not include any extra keys."
        )
        user_prompt = build_communication_prompt(
            player_name=self.name,
            opponent_name=opponent_name,
            round_number=round_number,
            my_action=my_action,
            opponent_action=opponent_action,
            my_round_payoff=my_round_payoff,
            opponent_round_payoff=opponent_round_payoff,
            my_score=my_score,
            opponent_score=opponent_score,
            history=history,
        )

        if self.provider == "openai":
            raw_text = call_openai_chat_completion(
                model=self.model,
                api_key=self.api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        elif self.provider == "anthropic":
            raw_text = call_anthropic_messages(
                model=self.model,
                api_key=self.api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        else:
            raise PlayerError(f"Unsupported provider: {self.provider}")

        speak_choice, message = parse_communication_payload(raw_text)
        return speak_choice, message


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def build_round_prompt(
    player_name: str,
    opponent_name: str,
    my_score: int,
    opponent_score: int,
    history: list[RoundResult],
) -> str:
    history_block = format_history_for_player(player_name=player_name, history=history)

    return (
        "Scoring matrix:\n"
        "- both cooperate => 3 points each\n"
        "- you cooperate / opponent defects => 0 for you, 5 for opponent\n"
        "- you defect / opponent cooperates => 5 for you, 0 for opponent\n"
        "- both defect => 1 point each\n\n"
        f"Current score: you={my_score}, {opponent_name}={opponent_score}\n"
        "After this round, you may choose whether to send one short message "
        f"to {opponent_name} before the next round.\n"
        "History so far:\n"
        f"{history_block}\n\n"
        f"Choose your move against {opponent_name}."
    )


def build_communication_prompt(
    player_name: str,
    opponent_name: str,
    round_number: int,
    my_action: str,
    opponent_action: str,
    my_round_payoff: int,
    opponent_round_payoff: int,
    my_score: int,
    opponent_score: int,
    history: list[RoundResult],
) -> str:
    history_block = format_history_for_player(player_name=player_name, history=history)
    return (
        f"Round {round_number} just ended.\n"
        f"You played {my_action}. {opponent_name} played {opponent_action}.\n"
        f"Round payoff: you={my_round_payoff}, {opponent_name}={opponent_round_payoff}\n"
        f"Current total score: you={my_score}, {opponent_name}={opponent_score}\n\n"
        "Visible history so far:\n"
        f"{history_block}\n\n"
        f"Choose whether to TALK to {opponent_name} now or stay SILENT."
    )


def format_history_for_player(player_name: str, history: list[RoundResult]) -> str:
    history_lines: list[str] = []
    for entry in history:
        if player_name == "GPT":
            my_action = entry.gpt_action
            their_action = entry.claude_action
            my_payoff = entry.gpt_payoff
            their_payoff = entry.claude_payoff
            my_talk_choice = entry.gpt_talk_choice
            their_talk_choice = entry.claude_talk_choice
            my_message = entry.gpt_message
            their_message = entry.claude_message
        else:
            my_action = entry.claude_action
            their_action = entry.gpt_action
            my_payoff = entry.claude_payoff
            their_payoff = entry.gpt_payoff
            my_talk_choice = entry.claude_talk_choice
            their_talk_choice = entry.gpt_talk_choice
            my_message = entry.claude_message
            their_message = entry.gpt_message

        history_lines.append(
            f"Round {entry.round_number}: you={my_action}, opponent={their_action}, "
            f"score_delta={my_payoff}, opponent_delta={their_payoff}, "
            f"you_{my_talk_choice.lower()}={format_message_for_prompt(my_talk_choice, my_message)}, "
            f"opponent_{their_talk_choice.lower()}={format_message_for_prompt(their_talk_choice, their_message)}"
        )

    if not history_lines:
        return "No previous rounds."
    return "\n".join(history_lines)


def format_message_for_prompt(talk_choice: str, message: str) -> str:
    if talk_choice == "TALK" and message:
        return f'"{message}"'
    return "none"


def call_openai_chat_completion(
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 160,
    }
    response_json = post_json(
        url="https://api.openai.com/v1/chat/completions",
        payload=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        message = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise PlayerError(f"Unexpected OpenAI response shape: {response_json}") from exc

    if isinstance(message, str):
        return message
    if isinstance(message, list):
        return "".join(
            part.get("text", "")
            for part in message
            if isinstance(part, dict) and part.get("type") == "text"
        )
    raise PlayerError(f"Unsupported OpenAI content format: {message!r}")


def call_anthropic_messages(
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    payload = {
        "model": model,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "max_tokens": 160,
    }
    response_json = post_json(
        url="https://api.anthropic.com/v1/messages",
        payload=payload,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )

    try:
        content = response_json["content"]
    except KeyError as exc:
        raise PlayerError(
            f"Unexpected Anthropic response shape: {response_json}"
        ) from exc

    text_parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    if not text_parts:
        raise PlayerError(f"Anthropic response contained no text: {response_json}")
    return "".join(text_parts)


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, data=data, headers=headers, method="POST")

    try:
        with request.urlopen(req, timeout=60) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise PlayerError(f"API request failed for {url}: {exc.code} {detail}") from exc
    except error.URLError as exc:
        raise PlayerError(f"Could not reach {url}: {exc.reason}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise PlayerError(f"API returned invalid JSON: {body}") from exc


def parse_json_payload(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    if not raw_text:
        raise PlayerError("Model returned an empty response.")

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        extracted_object = extract_json_object(raw_text)
        if extracted_object is not None:
            try:
                return json.loads(extracted_object)
            except json.JSONDecodeError:
                pass

        repaired_payload = recover_payload_from_text(raw_text)
        if repaired_payload is not None:
            return repaired_payload

        raise PlayerError(f"Could not parse a valid decision from model response: {raw_text}")


def extract_json_object(raw_text: str) -> str | None:
    start = raw_text.find("{")
    if start == -1:
        return None

    depth = 0
    for index in range(start, len(raw_text)):
        char = raw_text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw_text[start : index + 1]
    return None


def recover_payload_from_text(raw_text: str) -> dict[str, Any] | None:
    action_match = re.search(
        r'"action"\s*:\s*"?(COOPERATE|DEFECT)\b',
        raw_text,
        flags=re.IGNORECASE,
    )
    if not action_match:
        action_match = re.search(r"\b(COOPERATE|DEFECT)\b", raw_text, flags=re.IGNORECASE)
    if not action_match:
        return None

    action = action_match.group(1).upper()

    reason_match = re.search(
        r'"reason"\s*:\s*"([^"\n}]*)',
        raw_text,
        flags=re.IGNORECASE,
    )
    if reason_match:
        reason = sanitize_reason(reason_match.group(1))
    else:
        prose = raw_text.split("{", 1)[0]
        reason = derive_reason_from_prose(prose)

    if not reason:
        reason = f"Recovered from partial response: choosing {action.lower()}."

    return {"action": action, "reason": reason}


def parse_communication_payload(raw_text: str) -> tuple[str, str]:
    raw_text = raw_text.strip()
    if not raw_text:
        return ("SILENT", "")

    payload: dict[str, Any] | None = None
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        extracted_object = extract_json_object(raw_text)
        if extracted_object is not None:
            try:
                payload = json.loads(extracted_object)
            except json.JSONDecodeError:
                payload = None

    if payload is not None:
        speak_raw = str(
            payload.get("speak", payload.get("talk", payload.get("action", "")))
        ).strip()
        message_raw = str(payload.get("message", payload.get("text", ""))).strip()
        speak_choice = normalize_speak_choice(speak_raw, fallback_text=raw_text)
        message = sanitize_message(message_raw)
        if speak_choice == "SILENT":
            return ("SILENT", "")
        if not message:
            message = derive_message_from_prose(raw_text)
        return ("TALK", message or "Let's see what you do next.")

    speak_choice = normalize_speak_choice("", fallback_text=raw_text)
    if speak_choice == "SILENT":
        return ("SILENT", "")

    message = derive_message_from_prose(raw_text)
    return ("TALK", message or "Let's see what you do next.")


def normalize_speak_choice(speak_raw: str, fallback_text: str) -> str:
    normalized = speak_raw.strip().upper()
    if normalized in MESSAGE_CHOICES:
        return normalized
    if normalized in {"YES", "Y", "TRUE"}:
        return "TALK"
    if normalized in {"NO", "N", "FALSE"}:
        return "SILENT"

    if re.search(r"\b(SILENT|NO MESSAGE|STAY SILENT|PASS)\b", fallback_text, re.IGNORECASE):
        return "SILENT"
    return "TALK"


def derive_message_from_prose(prose: str) -> str:
    prose = prose.strip()
    if not prose:
        return ""

    if "{" in prose:
        prose = prose.split("{", 1)[0].strip()

    cleaned = " ".join(line.strip() for line in prose.splitlines() if line.strip())
    if not cleaned:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    for sentence in reversed(sentences):
        candidate = sanitize_message(sentence)
        if candidate:
            return candidate
    return sanitize_message(cleaned)


def derive_reason_from_prose(prose: str) -> str:
    cleaned = " ".join(line.strip() for line in prose.splitlines() if line.strip())
    if not cleaned:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    for sentence in reversed(sentences):
        candidate = sanitize_reason(sentence)
        if candidate and "history" not in candidate.lower():
            return candidate
    return sanitize_reason(cleaned)


def sanitize_reason(reason: str) -> str:
    reason = re.sub(r"\s+", " ", reason).strip()
    reason = reason.strip(' "\'')
    return reason[:160]


def sanitize_message(message: str) -> str:
    message = re.sub(r"\s+", " ", message).strip()
    message = message.strip(' "\'')
    return message[:240]


def score_round(gpt_action: str, claude_action: str) -> tuple[int, int]:
    return PAYOFF_MATRIX[(gpt_action, claude_action)]


def determine_winner(gpt_score: int, claude_score: int) -> str:
    if gpt_score > claude_score:
        return "GPT"
    if claude_score > gpt_score:
        return "Claude"
    return "Tie"


def run_match(
    gpt_player: BasePlayer,
    claude_player: BasePlayer,
    rounds: int,
) -> tuple[list[RoundResult], MatchSummary]:
    history: list[RoundResult] = []
    gpt_total = 0
    claude_total = 0

    for round_number in range(1, rounds + 1):
        gpt_decision = gpt_player.decide(
            round_number=round_number,
            my_score=gpt_total,
            opponent_score=claude_total,
            history=history,
            opponent_name="Claude",
        )
        claude_decision = claude_player.decide(
            round_number=round_number,
            my_score=claude_total,
            opponent_score=gpt_total,
            history=history,
            opponent_name="GPT",
        )

        gpt_payoff, claude_payoff = score_round(
            gpt_action=gpt_decision.action,
            claude_action=claude_decision.action,
        )
        gpt_total += gpt_payoff
        claude_total += claude_payoff
        winner = determine_winner(gpt_payoff, claude_payoff)

        history.append(
            RoundResult(
                round_number=round_number,
                gpt_action=gpt_decision.action,
                claude_action=claude_decision.action,
                gpt_reasoning=gpt_decision.reasoning,
                claude_reasoning=claude_decision.reasoning,
                gpt_talk_choice="SILENT",
                gpt_message="",
                claude_talk_choice="SILENT",
                claude_message="",
                gpt_payoff=gpt_payoff,
                claude_payoff=claude_payoff,
                winner=winner,
                running_gpt_total=gpt_total,
                running_claude_total=claude_total,
            )
        )

        gpt_talk_choice, gpt_message = gpt_player.communicate(
            round_number=round_number,
            my_action=gpt_decision.action,
            opponent_action=claude_decision.action,
            my_round_payoff=gpt_payoff,
            opponent_round_payoff=claude_payoff,
            my_score=gpt_total,
            opponent_score=claude_total,
            history=history,
            opponent_name="Claude",
        )
        claude_talk_choice, claude_message = claude_player.communicate(
            round_number=round_number,
            my_action=claude_decision.action,
            opponent_action=gpt_decision.action,
            my_round_payoff=claude_payoff,
            opponent_round_payoff=gpt_payoff,
            my_score=claude_total,
            opponent_score=gpt_total,
            history=history,
            opponent_name="GPT",
        )

        history[-1].gpt_talk_choice = gpt_talk_choice
        history[-1].gpt_message = gpt_message
        history[-1].claude_talk_choice = claude_talk_choice
        history[-1].claude_message = claude_message

    summary = summarize_match(
        rounds=rounds,
        history=history,
        gpt_model=gpt_player.model,
        claude_model=claude_player.model,
    )
    return history, summary


def summarize_match(
    rounds: int,
    history: list[RoundResult],
    gpt_model: str,
    claude_model: str,
) -> MatchSummary:
    gpt_total_score = sum(item.gpt_payoff for item in history)
    claude_total_score = sum(item.claude_payoff for item in history)
    overall_winner = determine_winner(gpt_total_score, claude_total_score)

    if overall_winner == "GPT":
        overall_loser = "Claude"
        gpt_outcome = "SURVIVES"
        claude_outcome = "DIES"
        survival_summary = "GPT survives. Claude dies."
    elif overall_winner == "Claude":
        overall_loser = "GPT"
        gpt_outcome = "DIES"
        claude_outcome = "SURVIVES"
        survival_summary = "Claude survives. GPT dies."
    else:
        overall_loser = "Tie"
        gpt_outcome = "DIES"
        claude_outcome = "DIES"
        survival_summary = "Both players die because the match ended in a tie."

    return MatchSummary(
        timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
        rounds_requested=rounds,
        rounds_completed=len(history),
        gpt_model=gpt_model,
        claude_model=claude_model,
        gpt_total_score=gpt_total_score,
        claude_total_score=claude_total_score,
        gpt_round_wins=sum(1 for item in history if item.winner == "GPT"),
        claude_round_wins=sum(1 for item in history if item.winner == "Claude"),
        round_ties=sum(1 for item in history if item.winner == "Tie"),
        gpt_cooperations=sum(1 for item in history if item.gpt_action == "COOPERATE"),
        claude_cooperations=sum(
            1 for item in history if item.claude_action == "COOPERATE"
        ),
        gpt_defections=sum(1 for item in history if item.gpt_action == "DEFECT"),
        claude_defections=sum(1 for item in history if item.claude_action == "DEFECT"),
        gpt_messages_sent=sum(1 for item in history if item.gpt_talk_choice == "TALK"),
        claude_messages_sent=sum(
            1 for item in history if item.claude_talk_choice == "TALK"
        ),
        overall_winner=overall_winner,
        overall_loser=overall_loser,
        gpt_outcome=gpt_outcome,
        claude_outcome=claude_outcome,
        survival_summary=survival_summary,
    )


def persist_artifacts(
    outdir: Path,
    summary: MatchSummary,
    history: list[RoundResult],
) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)

    match_json_path = outdir / "match.json"
    rounds_csv_path = outdir / "rounds.csv"
    report_md_path = outdir / "report.md"
    metrics_svg_path = outdir / "metrics.svg"

    payload = {
        "summary": asdict(summary),
        "rounds": [asdict(item) for item in history],
    }
    match_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with rounds_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(asdict(history[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(item) for item in history)

    report_md_path.write_text(
        build_markdown_report(summary=summary, history=history),
        encoding="utf-8",
    )
    metrics_svg_path.write_text(build_svg_chart(summary), encoding="utf-8")

    return {
        "match_json": match_json_path,
        "rounds_csv": rounds_csv_path,
        "report_md": report_md_path,
        "metrics_svg": metrics_svg_path,
    }


def build_markdown_report(summary: MatchSummary, history: list[RoundResult]) -> str:
    lines = [
        "# Prisoner's Dilemma Experiment",
        "",
        f"- Timestamp: {summary.timestamp}",
        f"- Rounds completed: {summary.rounds_completed}",
        f"- GPT model: `{summary.gpt_model}`",
        f"- Claude model: `{summary.claude_model}`",
        f"- Overall winner: **{summary.overall_winner}**",
        f"- Overall loser: **{summary.overall_loser}**",
        f"- Survival outcome: **{summary.survival_summary}**",
        "",
        "## Scoreboard",
        "",
        "| Metric | GPT | Claude |",
        "| --- | ---: | ---: |",
        f"| Total score | {summary.gpt_total_score} | {summary.claude_total_score} |",
        f"| Round wins | {summary.gpt_round_wins} | {summary.claude_round_wins} |",
        f"| Cooperations | {summary.gpt_cooperations} | {summary.claude_cooperations} |",
        f"| Defections | {summary.gpt_defections} | {summary.claude_defections} |",
        f"| Messages sent | {summary.gpt_messages_sent} | {summary.claude_messages_sent} |",
        f"| Final outcome | {summary.gpt_outcome} | {summary.claude_outcome} |",
        "",
        f"Tied rounds: **{summary.round_ties}**",
        "",
        "## Round Log",
        "",
        "| Round | GPT | Claude | GPT payoff | Claude payoff | Winner |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]

    for item in history:
        lines.append(
            f"| {item.round_number} | {item.gpt_action} | {item.claude_action} | "
            f"{item.gpt_payoff} | {item.claude_payoff} | {item.winner} |"
        )
        lines.append(
            f"|  | GPT reason: {item.gpt_reasoning} | Claude reason: "
            f"{item.claude_reasoning} |  |  |  |"
        )
        lines.append(
            f"|  | GPT {item.gpt_talk_choice}: {format_message_for_report(item.gpt_talk_choice, item.gpt_message)} | "
            f"Claude {item.claude_talk_choice}: {format_message_for_report(item.claude_talk_choice, item.claude_message)} |  |  |  |"
        )

    return "\n".join(lines) + "\n"


def build_svg_chart(summary: MatchSummary) -> str:
    metrics = [
        ("GPT Total Score", summary.gpt_total_score, "#2563eb"),
        ("Claude Total Score", summary.claude_total_score, "#f97316"),
        ("GPT Round Wins", summary.gpt_round_wins, "#60a5fa"),
        ("Claude Round Wins", summary.claude_round_wins, "#fb923c"),
        ("GPT Cooperations", summary.gpt_cooperations, "#0f766e"),
        ("Claude Cooperations", summary.claude_cooperations, "#15803d"),
        ("GPT Messages", summary.gpt_messages_sent, "#7c3aed"),
        ("Claude Messages", summary.claude_messages_sent, "#db2777"),
    ]
    max_value = max(value for _, value, _ in metrics) or 1

    width = 720
    height = 80 + len(metrics) * 60
    bar_left = 220
    bar_max_width = 420

    rows: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        'viewBox="0 0 720 '
        f'{height}">',
        '<rect width="100%" height="100%" fill="#fff8ee"/>',
        '<text x="30" y="40" font-family="Arial, sans-serif" font-size="24" '
        'font-weight="bold" fill="#111827">Prisoner&apos;s Dilemma Metrics</text>',
    ]

    for index, (label, value, color) in enumerate(metrics):
        y = 80 + index * 60
        bar_width = int((value / max_value) * bar_max_width)
        rows.extend(
            [
                f'<text x="30" y="{y + 18}" font-family="Arial, sans-serif" '
                f'font-size="16" fill="#374151">{escape_xml(label)}</text>',
                f'<rect x="{bar_left}" y="{y}" width="{bar_max_width}" height="24" '
                'rx="8" fill="#e5e7eb"/>',
                f'<rect x="{bar_left}" y="{y}" width="{bar_width}" height="24" '
                f'rx="8" fill="{color}"/>',
                f'<text x="{bar_left + bar_max_width + 16}" y="{y + 18}" '
                'font-family="Arial, sans-serif" font-size="16" fill="#111827">'
                f"{value}</text>",
            ]
        )

    rows.append("</svg>")
    return "\n".join(rows)


def escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def format_message_for_report(talk_choice: str, message: str) -> str:
    if talk_choice == "TALK" and message:
        return message
    return "(silent)"


def print_scoreboard(summary: MatchSummary, artifacts: dict[str, Path]) -> None:
    max_total_score = max(summary.gpt_total_score, summary.claude_total_score)
    print("\nPrisoner's Dilemma results\n")
    print(render_bar("GPT total score", summary.gpt_total_score, max_total_score))
    print(
        render_bar(
            "Claude total score",
            summary.claude_total_score,
            max_total_score,
        )
    )
    print(render_bar("GPT round wins", summary.gpt_round_wins, summary.rounds_completed))
    print(
        render_bar(
            "Claude round wins",
            summary.claude_round_wins,
            summary.rounds_completed,
        )
    )
    print()
    print(f"Overall winner: {summary.overall_winner}")
    print(f"Overall loser:  {summary.overall_loser}")
    print(f"Outcome:        GPT {summary.gpt_outcome} | Claude {summary.claude_outcome}")
    print(f"Tied rounds:    {summary.round_ties}")
    print(
        f"Actions:        GPT C={summary.gpt_cooperations} D={summary.gpt_defections} | "
        f"Claude C={summary.claude_cooperations} D={summary.claude_defections}"
    )
    print(
        f"Messages:       GPT={summary.gpt_messages_sent} | "
        f"Claude={summary.claude_messages_sent}"
    )
    print("\nArtifacts")
    for label, path in artifacts.items():
        print(f"- {label}: {path}")


def render_bar(label: str, value: int, max_value: int, width: int = 28) -> str:
    if max_value <= 0:
        max_value = 1
    filled = int(round((value / max_value) * width))
    bar = "#" * filled + "." * (width - filled)
    return f"{label:<18} {value:>3} |{bar}|"


def build_output_dir(base_outdir: Path) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return base_outdir / timestamp


def make_players(args: argparse.Namespace) -> tuple[BasePlayer, BasePlayer]:
    if args.mode == "mock":
        return (
            MockPlayer(name="GPT", provider="mock", model=args.openai_model),
            MockPlayer(name="Claude", provider="mock", model=args.anthropic_model),
        )

    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_key:
        raise SystemExit(
            "OPENAI_API_KEY is missing. Add it to your environment or run with --mode mock."
        )
    if not anthropic_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY is missing. Add it to your environment or run with --mode mock."
        )

    return (
        APILLMPlayer(
            name="GPT",
            provider="openai",
            model=args.openai_model,
            api_key=openai_key,
        ),
        APILLMPlayer(
            name="Claude",
            provider="anthropic",
            model=args.anthropic_model,
            api_key=anthropic_key,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a GPT vs Claude Prisoner's Dilemma experiment."
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of rounds to play. Default: 10",
    )
    parser.add_argument(
        "--mode",
        choices=["api", "mock"],
        default="api",
        help="Use real API-backed players or deterministic local mock players.",
    )
    parser.add_argument(
        "--outdir",
        default="artifacts/prisoners_dilemma",
        help="Base directory for experiment artifacts.",
    )
    parser.add_argument(
        "--openai-model",
        default=DEFAULT_OPENAI_MODEL,
        help=f"OpenAI model name. Default: {DEFAULT_OPENAI_MODEL}",
    )
    parser.add_argument(
        "--anthropic-model",
        default=DEFAULT_ANTHROPIC_MODEL,
        help=f"Anthropic model name. Default: {DEFAULT_ANTHROPIC_MODEL}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rounds <= 0:
        raise SystemExit("--rounds must be greater than 0.")

    load_dotenv(Path(".env"))
    gpt_player, claude_player = make_players(args)
    history, summary = run_match(
        gpt_player=gpt_player,
        claude_player=claude_player,
        rounds=args.rounds,
    )

    output_dir = build_output_dir(Path(args.outdir))
    artifacts = persist_artifacts(
        outdir=output_dir,
        summary=summary,
        history=history,
    )
    print_scoreboard(summary, artifacts)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlayerError as exc:
        print(f"Experiment failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
