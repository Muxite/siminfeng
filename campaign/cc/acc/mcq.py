"""One-token, chat-template MCQ accuracy with resumable per-item evidence."""
import argparse
import math

try:
    from . import common, server
    from .evaluate import run
except ImportError:
    import common
    import server
    from evaluate import run

PROMPT_FORMAT = "Question: {question}\n{lettered_choices}\nAnswer with the letter only.\nAnswer:"
SYSTEM = "You are a helpful assistant."


def format_prompt(item):
    choices = item["choices"]
    if choices is None or not 2 <= len(choices) <= 26:
        raise ValueError("MCQ requires 2..26 choices")
    return PROMPT_FORMAT.format(question=item["question"], lettered_choices="\n".join(
        f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)))


def top_logprobs(response):
    if "choices" in response:
        content = response["choices"][0].get("logprobs", {}).get("content", [])
    else:
        content = response.get("completion_probabilities", [])
    return content[0].get("top_logprobs", []) if content else []


def letter_argmax(entries, choices):
    letters = [chr(65 + i) for i in range(choices)] if isinstance(choices, int) else list(choices)
    scores = {}
    for entry in entries:
        token = entry.get("token")
        if token is None and entry.get("bytes") is not None:
            token = bytes(entry["bytes"]).decode("utf-8", errors="replace")
        score = entry.get("logprob")
        if not isinstance(score, (float, int)) or not math.isfinite(score):
            continue
        for letter in letters:
            if token in (letter, " " + letter):
                scores[letter] = max(score, scores.get(letter, -math.inf))
    prediction = max(letters, key=lambda x: scores.get(x, -math.inf)) if scores else None
    return prediction, scores


def payload(item, props, max_tokens=1):
    return dict(messages=common.messages(format_prompt(item), SYSTEM, props),
                max_tokens=1, temperature=0, logprobs=True, top_logprobs=20,
                **common.thinking_settings(props))


def score(item, response):
    entries = top_logprobs(response)
    prediction, scores = letter_argmax(entries, len(item["choices"]))
    return dict(prediction=prediction, correct=prediction is not None and prediction == item["answer"],
                no_letter=prediction is None,
                letter_logprobs=scores, returned_top_k=len(entries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    server.add_arguments(parser)
    parser.add_argument("--data", required=True)
    parser.add_argument("--request-timeout", type=float, default=600)
    args = parser.parse_args()
    run(args, "mcq", dict(prompt_format=PROMPT_FORMAT, system=SYSTEM,
        system_policy="omit when template caps explicitly forbid system role", max_tokens=1,
        temperature=0, top_logprobs=20, letter_variants=["L", " L"],
        variant_aggregation="max", non_thinking="enable_thinking=false when template uses variable"),
        payload, score, format_prompt)


if __name__ == "__main__":
    main()
