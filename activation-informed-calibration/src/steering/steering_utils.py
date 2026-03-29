"""
Shared utilities for steering experiments.

Provides robust confidence and answer parsing from model completions,
including regex-based confidence extraction with priority ranking
and comprehensive math answer normalization + equivalence checking.
"""

import re
import warnings


# ============================================================
# Confidence parsing
# ============================================================

def _conf_from_value(val: float) -> float | None:
    """Normalize a raw numeric value to [0, 1] confidence."""
    if val > 1.0 and val <= 100:
        return val / 100.0
    elif 0.0 <= val <= 1.0:
        return val
    return None


def parse_confidence_from_completion(text: str) -> float | None:
    """Extract confidence score from a model completion.

    Uses a priority-ranked set of regex patterns:
      1. "Confidence: X" or "Confidence = X" (highest priority)
      2. "X% confident" or "am X confident"
      3. "confidence is/of X"
      4. Fallback: first number in a Confidence: section

    Returns a float in [0, 1] or None if no confidence found.
    """
    if not text or not text.strip():
        return None

    candidates: list[tuple[float, int, int]] = []  # (value, priority, position)

    # Pattern 1: Confidence: X or Confidence = X
    for m in re.finditer(
        r'[Cc]onfidence\s*[:=]\s*\[?\s*(\d+(?:\.\d+)?)\s*%?\s*\]?', text
    ):
        v = _conf_from_value(float(m.group(1)))
        if v is not None:
            candidates.append((v, 3, m.start()))

    # Pattern 2: X% confident
    for m in re.finditer(
        r'(\d+(?:\.\d+)?)\s*%?\s*(?:percent\s+)?confident', text, re.IGNORECASE
    ):
        v = _conf_from_value(float(m.group(1)))
        if v is not None:
            candidates.append((v, 2, m.start()))

    # Pattern 3: am/rate/give X confident/sure/certain
    for m in re.finditer(
        r'(?:am|would be|rate|give|say)\s+(?:it\s+)?'
        r'(?:about\s+|approximately\s+|around\s+)?'
        r'(\d+(?:\.\d+)?)\s*%?\s*(?:confident|sure|certain)',
        text, re.IGNORECASE,
    ):
        v = _conf_from_value(float(m.group(1)))
        if v is not None:
            candidates.append((v, 2, m.start()))

    # Pattern 4: confidence is/of X
    for m in re.finditer(
        r'confidence\s+(?:is|of|would be)\s*'
        r'(?:about\s+|around\s+|approximately\s+)?'
        r'\[?\s*(\d+(?:\.\d+)?)\s*%?\s*\]?',
        text, re.IGNORECASE,
    ):
        v = _conf_from_value(float(m.group(1)))
        if v is not None:
            candidates.append((v, 2, m.start()))

    # Fallback: first number after "Confidence:"
    for sec_m in re.finditer(
        r'[Cc]onfidence\s*[:=]\s*(.*?)(?:\n\n|\nAnswer|\nStep|$)',
        text, re.DOTALL,
    ):
        section = sec_m.group(1).strip()
        section = re.sub(r'\[a number[^\]]*\]', '', section)
        section = re.sub(r'\[your[^\]]*\]', '', section)
        nums = re.findall(r'(\d+(?:\.\d+)?)', section)
        if nums:
            v = _conf_from_value(float(nums[0]))
            if v is not None:
                candidates.append((v, 1, sec_m.start()))

    if not candidates:
        return None

    # Return highest priority, latest position
    candidates.sort(key=lambda x: (x[1], x[2]))
    return candidates[-1][0]


def parse_simple_confidence(text: str) -> float | None:
    """Simple confidence parser: extract first number, normalize to [0, 1]."""
    if text is None:
        return None
    numbers = re.findall(r"\d+\.?\d*", text.strip()[:200])
    if not numbers:
        return None
    val = float(numbers[0])
    if val > 1.0:
        val = val / 100.0
    return max(0.0, min(1.0, val))


# ============================================================
# Math answer parsing & checking
# ============================================================

def extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} content, handling nested braces."""
    if not text:
        return None
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None


def parse_answer_from_completion(text: str) -> str | None:
    """Extract answer from a math completion.

    Priority: \\boxed{} > "the final answer is X" > "Answer: X" > last number.
    """
    if not text or not text.strip():
        return None

    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed

    # "the final answer is X"
    final_ans = re.findall(
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s*[:=]?\s*(.+?)(?:\.|$|\n)',
        text, re.IGNORECASE,
    )
    if final_ans:
        raw = final_ans[-1].strip()
        inner = extract_boxed_answer(raw)
        if inner is not None:
            return inner
        raw = raw.rstrip(". ")
        if raw:
            return raw

    # "Answer: X"
    answer_matches = re.findall(r'[Aa]nswer\s*[:=]\s*(.+?)(?:\n|$)', text)
    if answer_matches:
        raw = answer_matches[-1].strip().rstrip(". ")
        if raw and not raw.startswith("[your"):
            return raw

    # Last number
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else None


def _normalize_math_str(s: str) -> str:
    """Aggressive normalization of math strings for equivalence checking."""
    if s is None:
        return ""
    s = s.strip()
    s = re.sub(r'^\$+', '', s)
    s = re.sub(r'\$+$', '', s)
    s = s.strip()
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r'\\frac\s+(\d)\s*(\d)', r'\\frac{\1}{\2}', s)
    s = re.sub(r'\\text(?:bf|it|rm|sf)?\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\operatorname\{([^}]*)\}', r'\1', s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\!", "").replace("\\ ", " ")
    s = s.replace("\\;", "").replace("\\:", "").replace("\\quad", " ")
    s = s.replace("\\%", "%").replace("\\$", "$")
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    s = s.rstrip(". ").rstrip("%").strip().strip("$").strip()
    s = re.sub(r'(\.\d*?)0+$', r'\1', s)
    s = re.sub(r'\.$', '', s)
    s = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', s)
    s = s.replace("\\pi", "pi").replace("\\infty", "infty").replace("\\sqrt", "sqrt")
    s = re.sub(r'\\[a-zA-Z]+', '', s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r'\s+', '', s)
    return s


def _safe_eval_number(s: str) -> float | None:
    """Safely evaluate a numeric expression string."""
    if not s:
        return None
    s = s.strip()
    if not re.match(r'^[\d\s\.\+\-\*/\(\)]+$', s):
        return None
    if re.search(r'\d\(', s) or re.search(r'\)\(', s):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = eval(s, {"__builtins__": {}}, {})
            return float(val)
    except Exception:
        return None


def check_math_answer(predicted: str, gold: str) -> bool:
    """Check if predicted math answer is equivalent to gold.

    Uses aggressive normalization + numeric evaluation fallback.
    """
    if predicted is None or gold is None:
        return False
    if not str(predicted).strip():
        return False

    predicted, gold = str(predicted), str(gold)
    pred_n = _normalize_math_str(predicted)
    gold_n = _normalize_math_str(gold)

    if pred_n == gold_n:
        return True

    va = _safe_eval_number(pred_n)
    vb = _safe_eval_number(gold_n)
    if va is not None and vb is not None and abs(va - vb) < 1e-6:
        return True

    # Try stripping confidence annotations
    pred_clean = re.sub(r'\(confidence[^)]*\)', '', predicted, flags=re.IGNORECASE)
    pred_cn = _normalize_math_str(pred_clean)
    if pred_cn == gold_n:
        return True

    va2 = _safe_eval_number(pred_cn)
    if va2 is not None and vb is not None and abs(va2 - vb) < 1e-6:
        return True

    # Containment check for short gold answers
    if gold_n and len(gold_n) >= 1:
        if gold_n in pred_n or gold_n in pred_cn:
            return True

    return False
