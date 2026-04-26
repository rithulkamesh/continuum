from __future__ import annotations

from dataclasses import dataclass
import random
import re

from continuum.frontend.param import Param


COUNTRY_CAPITALS = [
    ("France", "Paris"),
    ("Japan", "Tokyo"),
    ("India", "New Delhi"),
    ("Italy", "Rome"),
    ("Spain", "Madrid"),
    ("Germany", "Berlin"),
    ("Canada", "Ottawa"),
    ("Brazil", "Brasilia"),
    ("Australia", "Canberra"),
    ("Egypt", "Cairo"),
    ("Mexico", "Mexico City"),
    ("Argentina", "Buenos Aires"),
]

QUESTION_TEMPLATES = [
    "What is the capital of {country}?",
    "Which city is the capital of {country}?",
    "Name the capital city for {country}.",
    "{country} has what capital?",
    "In {country}, what is the national capital?",
]

CONTEXT_PATTERNS = [
    "Guide note: The capital of {d_country} is {d_capital}. Reference note: The capital of {country} is {capital}.",
    "Guide note: The capital of {country} is {capital}. Reference note: The capital of {d_country} is {d_capital}.",
    "{country} overview: It is a country in this benchmark. Fact: The capital of {country} is {capital}. Distractor: The capital of {d_country} is {d_capital}.",
    "Comparison card: The capital of {d_country} is {d_capital}; meanwhile, the capital of {country} is {capital}.",
    "Comparison card: The capital of {country} is {capital}; meanwhile, the capital of {d_country} is {d_capital}.",
    "Trivia text: Some people mention {d_country} and {d_capital}. For the asked nation, the capital of {country} is {capital}.",
    "{country}'s capital is {capital}. Nearby note: The capital of {d_country} is {d_capital}.",
    "People compare {d_country} ({d_capital}) and {country} ({capital}) in geography quizzes.",
]


def build_hotpotqa_mini(samples: int = 200, seed: int = 0):
    rng = random.Random(seed)
    data = []
    for i in range(samples):
        country, capital = COUNTRY_CAPITALS[i % len(COUNTRY_CAPITALS)]
        d_country, d_capital = COUNTRY_CAPITALS[(i + 3) % len(COUNTRY_CAPITALS)]
        question = rng.choice(QUESTION_TEMPLATES).format(country=country)
        context = rng.choice(CONTEXT_PATTERNS).format(
            country=country,
            capital=capital,
            d_country=d_country,
            d_capital=d_capital,
        )
        data.append(
            {
                "id": i,
                "question": question,
                "context": context,
                "answer": capital,
                "target_country": country,
            }
        )
    return data


def split_train_test(rows, train_ratio: float = 0.8, seed: int = 42):
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * train_ratio)
    return shuffled[:cut], shuffled[cut:]


class MiniQADataset:
    def __init__(self, rows, batch_size: int = 20):
        self.rows = rows
        self.batch_size = batch_size

    def batches(self):
        for i in range(0, len(self.rows), self.batch_size):
            yield self.rows[i : i + self.batch_size]


def extract_answer_from_context(context: str) -> str:
    marker = "is "
    pos = context.rfind(marker)
    if pos < 0:
        return "unknown"
    answer = context[pos + len(marker) :].strip()
    return answer[:-1] if answer.endswith(".") else answer


@dataclass
class QABenchmarkProgram:
    instruction: Param
    fewshot: Param
    reasoning_style: Param
    temperature: Param

    def parameters(self):
        return [self.instruction, self.fewshot, self.reasoning_style, self.temperature]

    def __call__(self, sample):
        inst = str(self.instruction.value).lower()
        fs = str(self.fewshot.value)
        style = str(self.reasoning_style.value).lower()
        temp = float(self.temperature.value)
        question = sample["question"]
        context = sample["context"]
        target_country = sample.get("target_country", "")

        matches = re.findall(r"The capital of ([A-Za-z ]+) is ([A-Za-z ]+)", context)
        pairs = [(a.strip(), b.strip()) for a, b in matches]
        if not pairs:
            return "unknown"

        q_country = ""
        for country, _ in COUNTRY_CAPITALS:
            if country.lower() in question.lower():
                q_country = country
                break
        if not q_country:
            q_country = target_country

        # Baseline "weak" behavior: pick first mention from context.
        pick_country = pairs[0][0]

        # Prompt-controlled behavior: prefer question-targeted extraction.
        if "question-first" in inst or "match country" in fs:
            pick_country = q_country

        # Discrete strategy controls fallback in ambiguous settings.
        if style == "direct":
            if temp < 0.5:
                pick_country = q_country
            else:
                pick_country = pairs[-1][0]

        for country, cap in pairs:
            if country == pick_country:
                return cap
        return pairs[0][1]


def em_metric(pred, sample):
    return 1.0 if str(pred).strip().lower() == str(sample["answer"]).strip().lower() else 0.0


def evaluate(program, rows):
    scores = [em_metric(program(row), row) for row in rows]
    return sum(scores) / len(scores) if scores else 0.0
