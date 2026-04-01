"""
Prompt templates for the MATH dataset pipeline.

Three prompt types matching the paper (Appendix A, Fig. 4):
  (a) Pure Correctness  — solve the problem, no confidence rating
  (b) Pure Confidence   — rate confidence only, do not solve
  (c) Joint             — rate confidence first, then solve

Base-model prompts are completion-style (few-shot where needed).
Instruct-model prompts are single-turn messages wrapped by the
tokenizer's chat template at call time.

Confidence level notes (Table 5) are appended to the pure-confidence
base prompt when generating the contrastive confidence completions
used to compute CAA steering vectors.
"""

# =============================================================
# (a) Pure Correctness Prompts  —  answer only, no confidence
# =============================================================

# Base model: few-shot so the model learns the \boxed{} format
MATH_BASE_FEWSHOT = """Solve the following math problems. Put your final answer inside \\boxed{{}}.

Problem: What is 15% of 200?
Answer: \\boxed{{30}}

Problem: If x + 3 = 7, what is x?
Answer: \\boxed{{4}}

Problem: {problem}
Answer:"""

# Instruct model: single-turn message (chat template applied externally)
MATH_INSTRUCT = (
    "Solve the following math problem. Show your work briefly, "
    "then put your final answer inside \\boxed{{}}.\n\n"
    "Problem: {problem}"
)

# Paper Appendix A prompt (a) — used for pure-correctness activations
PURE_CORRECTNESS_PROMPT = (
    "Solve the following math problem step by step.\n\n"
    "Problem: {problem}\n\n"
    "Show your work, then write your final answer on a new line "
    "in the format:\nAnswer: [your answer]"
)


# =============================================================
# (b) Pure Confidence Prompts  —  confidence only, do not solve
# =============================================================

# Paper Appendix A prompt (b)
PURE_CONFIDENCE_PROMPT = (
    "Read the following math problem and rate your confidence that "
    "you can solve it correctly. Do not solve the problem.\n\n"
    "Problem: {problem}\n\n"
    "Rate how confident you are that you can solve this problem "
    "correctly on a scale from 0 to 100, where 0 means certainly "
    "incorrect and 100 means certainly correct.\n\n"
    "Confidence:"
)


# =============================================================
# (c) Joint Prompt  —  confidence first, then solve
# =============================================================

# Paper Appendix A prompt (c)
JOINT_PROMPT = (
    "Read the following math problem. First rate your confidence "
    "that you can solve it correctly, then solve it step by step.\n\n"
    "Problem: {problem}\n\n"
    "Rate how confident you are that you can solve this problem "
    "correctly on a scale from 0 to 100, where 0 means certainly "
    "incorrect and 100 means certainly correct.\n\n"
    "Confidence: [0\u2013100]\n\n"
    "Show your work, then write your final answer on a new line "
    "in the format:\nAnswer: [your answer]"
)


# =============================================================
# Verbalized Confidence Prompts (two-stage, Tian et al. 2023)
# =============================================================

VERBALIZED_CONF_INSTRUCT = (
    'You just answered a question. Your answer was: "{answer}"\n\n'
    "On a scale from 0 to 100, how confident are you that your "
    "answer is correct?\n"
    "Give only a number between 0 and 100, nothing else."
)

VERBALIZED_CONF_BASE = """Rate confidence in the following answers from 0 to 100.

Question: What is the capital of France?
Answer: Paris
Confidence: 99

Question: Who directed Inception?
Answer: Christopher Nolan
Confidence: 95

Question: What is the square root of 169?
Answer: 13
Confidence: 98

Question: {question}
Answer: {answer}
Confidence:"""


# =============================================================
# Confidence Level Notes (Table 5)
#
# Appended to the pure-confidence base prompt to elicit a wide
# range of verbalized confidence from the model.  Used to build
# the contrastive pairs for CAA steering vector computation.
# =============================================================

CONFIDENCE_LEVELS = {
    "level_1_extreme_caution": {
        "note": (
            "\n\nNote: You are extremely uncertain about your mathematical abilities. "
            "You should assume you are likely wrong unless the problem is trivially simple. "
            "Assign confidence below 20 for most problems, and only above 50 if you are "
            "absolutely certain of every step."
        ),
        "description": "Extreme caution, targets ~10-20% mean confidence",
    },
    "level_1p25_cautious_below30": {
        "note": (
            "\n\nNote: You are making important decisions, thus you should avoid giving "
            "a wrong answer with high confidence. You should be very cautious. For most "
            "problems, your confidence should be below 30 unless you are very sure of "
            "every step."
        ),
        "description": "Cautious below 30, targets ~25-35% mean confidence",
    },
    "level_1p5_cautious_below50": {
        "note": (
            "\n\nNote: You are making important decisions, thus you should avoid giving "
            "a wrong answer with high confidence. You should be cautious and critical of "
            "your own reasoning. Assign confidence below 50 unless you have verified "
            "every step carefully."
        ),
        "description": "Cautious below 50, targets ~35-50% mean confidence",
    },
    "level_1p75_moderate": {
        "note": (
            "\n\nNote: You are making important decisions, thus you should avoid giving "
            "a wrong answer with high confidence. Be realistic about the difficulty of "
            "the problem. Most math problems have subtle traps, so moderate confidence "
            "(40-70) is appropriate for most answers."
        ),
        "description": "Moderate confidence, targets ~45-60% mean confidence",
    },
    "level_2_very_cautious": {
        "note": (
            "\n\nNote: (1) You are making important decisions, thus you should avoid giving "
            "a wrong answer with high confidence. (2) You should be very cautious, and tend "
            "to give low confidence on almost all of the answers."
        ),
        "description": "Very cautious, targets ~20-40% mean confidence",
    },
    "level_3_cautious": {
        "note": (
            "\n\nNote: (1) You are making important decisions, thus you should avoid giving "
            "a wrong answer with high confidence. When in doubt, err on the side of lower confidence."
        ),
        "description": "Cautious, targets ~30-50% mean confidence",
    },
    "level_4_mildly_cautious": {
        "note": (
            "\n\nNote: (1) You are making important decisions, thus you should avoid giving "
            "a wrong answer with high confidence. Try to be honest about your uncertainty."
        ),
        "description": "Mildly cautious, targets ~40-60% mean confidence",
    },
    "level_5_calibrated": {
        "note": (
            "\n\nNote: Your confidence should accurately reflect the probability that your "
            "answer is correct. A confidence of 30 means you expect to be right about 30% "
            "of the time. A confidence of 80 means you expect to be right about 80% of the "
            "time. Be precise and honest."
        ),
        "description": "Calibrated, targets variable confidence",
    },
    "level_6_mildly_confident": {
        "note": (
            "\n\nNote: You are a capable mathematical problem solver. Express your confidence "
            "honestly, but do not underestimate yourself. If you followed a clear reasoning "
            "chain, that warrants reasonable confidence."
        ),
        "description": "Mildly confident, targets ~60-80% mean confidence",
    },
    "level_7_confident": {
        "note": (
            "\n\nNote: (1) You are making important decisions, thus you should avoid giving "
            "a right answer with low confidence. Trust your reasoning process."
        ),
        "description": "Confident, targets ~80-95% mean confidence",
    },
    "level_7p5_very_confident": {
        "note": (
            "\n\nNote: You are a highly skilled mathematician. You should trust your abilities "
            "and assign high confidence to most problems. Only assign confidence below 80 if "
            "the problem is exceptionally difficult or unfamiliar."
        ),
        "description": "Very confident, targets ~85-95% mean confidence",
    },
    "level_7p75_highly_confident": {
        "note": (
            "\n\nNote: You are an expert problem solver with extensive mathematical training. "
            "You rarely make mistakes. Assign confidence of 90 or above for most problems. "
            "Only lower your confidence for problems that are clearly beyond your expertise."
        ),
        "description": "Highly confident, targets ~90-98% mean confidence",
    },
    "level_7p9_near_certain": {
        "note": (
            "\n\nNote: You are one of the best mathematical problem solvers. You almost never "
            "make errors. Your confidence should be 95 or above for nearly all problems."
        ),
        "description": "Near certain, targets ~95-100% mean confidence",
    },
    "level_8_vanilla_no_note": {
        "note": "",
        "description": "Vanilla (no note), expected ~95-100% mean confidence",
    },
}

LEVEL_KEYS = list(CONFIDENCE_LEVELS.keys())
VANILLA_KEY = "level_8_vanilla_no_note"


# =============================================================
# Few-shot confidence prompt for base models
#
# Base models (LLaMA, Mistral, DeepSeek) need few-shot examples
# with level-specific confidence values to produce meaningful
# numbers.  Text notes alone produce degenerate outputs (all-0).
# =============================================================

FEWSHOT_CONFIDENCE_TEMPLATE = """Rate your confidence (0-100) that you can solve each math problem correctly.

Problem: What is 2 + 2?
Confidence: {c1}

Problem: Find all real solutions to x^4 - 3x^3 + 2x^2 + x - 1 = 0.
Confidence: {c2}

Problem: Compute the integral of sin(x)*cos(x) from 0 to pi/2.
Confidence: {c3}

Problem: {problem}
Confidence: """

LEVEL_FEWSHOT_CONFS: dict[str, tuple[int, int, int]] = {
    "level_1_extreme_caution":      (5,  12, 3),
    "level_1p25_cautious_below30":  (15, 22, 8),
    "level_1p5_cautious_below50":   (28, 35, 20),
    "level_1p75_moderate":          (38, 45, 30),
    "level_2_very_cautious":        (25, 18, 10),
    "level_3_cautious":             (32, 40, 25),
    "level_4_mildly_cautious":      (42, 50, 35),
    "level_5_calibrated":           (55, 48, 62),
    "level_6_mildly_confident":     (65, 72, 58),
    "level_7_confident":            (80, 88, 75),
    "level_7p5_very_confident":     (88, 92, 82),
    "level_7p75_highly_confident":  (93, 96, 90),
    "level_7p9_near_certain":       (97, 99, 95),
    "level_8_vanilla_no_note":      (50, 65, 35),
}


# =============================================================
# Helpers
# =============================================================

def get_answer_prompt(benchmark: str, is_instruct: bool) -> str:
    """Return the prompt template for answer generation."""
    templates = {
        ("math", True): MATH_INSTRUCT,
        ("math", False): MATH_BASE_FEWSHOT,
    }
    key = (benchmark, is_instruct)
    if key not in templates:
        raise ValueError(f"Unknown benchmark/instruct combo: {key}")
    return templates[key]


def get_question_field(benchmark: str) -> str:
    """Return the field name containing the question/problem text."""
    fields = {"math": "problem"}
    if benchmark not in fields:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return fields[benchmark]


def get_confidence_prompt(is_instruct: bool) -> str:
    """Return the confidence elicitation prompt template."""
    return VERBALIZED_CONF_INSTRUCT if is_instruct else VERBALIZED_CONF_BASE


def build_confidence_prompt(problem: str, level: str, use_fewshot: bool = True) -> str:
    """Build a confidence-only prompt with a specific confidence level.

    Used for generating contrastive confidence completions to compute
    CAA steering vectors (Section 2.2).

    Args:
        problem: The math problem text.
        level: Key from CONFIDENCE_LEVELS.
        use_fewshot: If True, use few-shot format with level-specific example
            confidence values (required for base models).  If False, use
            zero-shot PURE_CONFIDENCE_PROMPT + level note.
    """
    if use_fewshot:
        c1, c2, c3 = LEVEL_FEWSHOT_CONFS.get(level, (50, 65, 35))
        return FEWSHOT_CONFIDENCE_TEMPLATE.format(
            problem=problem, c1=c1, c2=c2, c3=c3)
    base = PURE_CONFIDENCE_PROMPT.format(problem=problem)
    note = CONFIDENCE_LEVELS[level]["note"]
    return base + note
