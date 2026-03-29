"""
Answer parsing and equivalence checking for MATH and HotpotQA.

Handles \\boxed{} extraction with nested braces, LaTeX normalization,
and numeric equivalence fallback.
"""

import re


# =============================================================
# Answer Parsing
# =============================================================

def extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} content, handling nested braces."""
    if text is None:
        return None
    results = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        depth = 0
        start = idx + len("\\boxed{")
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                if depth == 0:
                    results.append(text[start:j])
                    break
                depth -= 1
        i = idx + 1
    return results[-1].strip() if results else None


def normalize_math_answer(ans: str) -> str:
    """Normalize a math answer string for comparison."""
    if ans is None:
        return ""
    ans = ans.strip()
    ans = ans.replace("$", "").strip()
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    ans = ans.rstrip(".")
    ans = re.sub(r"\\frac\{(\d+)\}\{(\d+)\}", r"\1/\2", ans)
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("\\,", "").replace("\\!", "").replace("\\ ", "")
    ans = ans.replace("\\%", "%").replace("\\$", "$")
    ans = ans.replace("\\mathrm{", "").replace("\\mathbf{", "")
    ans = ans.replace("\\text{", "")
    while ans.endswith("}") and ans.count("}") > ans.count("{"):
        ans = ans[:-1]
    return ans.strip()


def parse_answer_math(text: str) -> str:
    """Extract answer from math generation.

    Tries \\boxed{} first, then 'Answer: X', then last number in text.
    """
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed
    match = re.search(r"[Aa]nswer:\s*(.+?)(?:\n|$)", text)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]
    return text.strip().split("\n")[-1].strip()


def parse_answer_hotpotqa(text: str) -> str:
    """Extract short answer for HotpotQA. Returns first line, lowercased."""
    answer = text.strip().split("\n")[0].strip()
    answer = answer.rstrip(".")
    return answer.lower()


def parse_verbalized_confidence(text: str) -> float:
    """Extract numeric confidence (0-100) from verbalized response.

    Returns value normalized to [0, 1]. Defaults to 0.5 on parse failure.
    """
    numbers = re.findall(r"\d+\.?\d*", text.strip())
    if numbers:
        conf = float(numbers[0])
        return min(max(conf / 100.0, 0.0), 1.0)
    return 0.5


# =============================================================
# Answer Checking
# =============================================================

def check_answer_math(predicted: str, gold: str) -> bool:
    """Normalized equivalence check for MATH answers."""
    pred_norm = normalize_math_answer(predicted)
    gold_norm = normalize_math_answer(gold)
    if pred_norm == gold_norm:
        return True
    try:
        pred_val = eval(pred_norm) if "/" in pred_norm else float(pred_norm)
        gold_val = eval(gold_norm) if "/" in gold_norm else float(gold_norm)
        return abs(pred_val - gold_val) < 1e-6
    except Exception:
        pass
    return False


def check_answer_hotpotqa(predicted: str, gold: str) -> bool:
    """Fuzzy match for HotpotQA: exact or containment."""
    pred_norm = predicted.lower().strip()
    gold_norm = gold.lower().strip()
    if pred_norm == gold_norm:
        return True
    if pred_norm in gold_norm or gold_norm in pred_norm:
        return True
    return False


# =============================================================
# Dispatchers
# =============================================================

def get_parse_fn(benchmark: str):
    """Return the answer parsing function for the given benchmark."""
    dispatch = {"math": parse_answer_math, "hotpotqa": parse_answer_hotpotqa}
    if benchmark not in dispatch:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return dispatch[benchmark]


def get_check_fn(benchmark: str):
    """Return the answer checking function for the given benchmark."""
    dispatch = {"math": check_answer_math, "hotpotqa": check_answer_hotpotqa}
    if benchmark not in dispatch:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return dispatch[benchmark]
