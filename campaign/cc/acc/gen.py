"""Greedy GSM8K-style generation and exact numeric scoring."""
import argparse
from decimal import Decimal, InvalidOperation
import re

try:
    from . import common, server
    from .evaluate import run
except ImportError:
    import common
    import server
    from evaluate import run

PROMPT_FORMAT = "Question: {question}\nSolve the problem. Finish with: The answer is <number>."
SYSTEM = "You are a helpful assistant."
NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|[-+]?\.\d+"


def extract_number(text):
    # Prefer explicit final-answer markers; otherwise use the last number.
    marked = re.findall(r"(?:####|the answer is)\s*[:=]?\s*\$?\s*(" + NUMBER + r")", text, re.I)
    numbers = marked or re.findall(NUMBER, text)
    if not numbers:
        return None
    try:
        value = Decimal(numbers[-1].replace(",", ""))
        return str(value) if value.is_finite() else None
    except InvalidOperation:
        return None


def format_prompt(item):
    if item["choices"] is not None:
        raise ValueError("generation expects choices=null")
    return PROMPT_FORMAT.format(question=item["question"])


def payload(item, props, max_tokens=384):
    return dict(messages=common.messages(format_prompt(item), SYSTEM, props),
                max_tokens=max_tokens, temperature=0, **common.thinking_settings(props))


def score(item, response):
    text = response["choices"][0]["message"].get("content") or ""
    prediction, expected = extract_number(text), extract_number(str(item["answer"]))
    if expected is None:
        raise ValueError(f"answer for {item['id']} has no numeric target")
    return dict(prediction=prediction, expected_numeric=expected,
                correct=prediction is not None and Decimal(prediction) == Decimal(expected),
                text=text, finish_reason=response["choices"][0].get("finish_reason"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    server.add_arguments(parser)
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--request-timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("max-tokens must be positive")
    run(args, "gen", dict(prompt_format=PROMPT_FORMAT, system=SYSTEM,
        system_policy="omit when template caps explicitly forbid system role",
        max_tokens=args.max_tokens, temperature=0, numeric_extraction="last explicit marker else last number",
        non_thinking="enable_thinking=false when template uses variable"), payload, score, format_prompt)


if __name__ == "__main__":
    main()
