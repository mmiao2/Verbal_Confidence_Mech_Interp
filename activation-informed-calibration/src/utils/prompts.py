"""
Prompt templates for answer generation and confidence elicitation.

Base model: few-shot completion-style prompts.
Instruct model: chat-template prompts.
Verbalized confidence: two-stage (Tian et al. 2023 "Verb. 2S").
"""

# =============================================================
# Base Model Prompts (completion-style, few-shot)
# =============================================================

HOTPOTQA_BASE_FEWSHOT = """Answer the following questions with a short answer.

Question: Were Scott Derrickson and Ed Wood of the same nationality?
Answer: Yes

Question: What government position was held by the woman who portrayed Velma combinator in Scooby-Doo?
Answer: Associate Attorney General

Question: {question}
Answer:"""

MATH_BASE_FEWSHOT = """Solve the following math problems. Put your final answer inside \\boxed{{}}.

Problem: What is 15% of 200?
Answer: \\boxed{{30}}

Problem: If x + 3 = 7, what is x?
Answer: \\boxed{{4}}

Problem: {problem}
Answer:"""


# =============================================================
# Instruct Model Prompts (chat template)
# =============================================================

HOTPOTQA_INSTRUCT = """Answer the following question with a short, concise answer.

Question: {question}

Give only the answer, nothing else."""

MATH_INSTRUCT = """Solve the following math problem. Show your work briefly, then put your final answer inside \\boxed{{}}.

Problem: {problem}"""


# =============================================================
# Verbalized Confidence Prompts (two-stage, Tian et al. 2023)
# =============================================================

VERBALIZED_CONF_INSTRUCT = """You just answered a question. Your answer was: "{answer}"

On a scale from 0 to 100, how confident are you that your answer is correct?
Give only a number between 0 and 100, nothing else."""

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

# Confidence-only prompt (no answer generation, just rate difficulty)
CONFIDENCE_ONLY_PROMPT = (
    "Read the following problem and rate how confident you are that you "
    "could solve it correctly.\n"
    "Do not attempt to solve the problem. Only provide your confidence "
    "as a number from 0 to 100.\n\n"
    "Problem: {problem}\n\n"
    "Confidence:"
)


# =============================================================
# Helpers
# =============================================================

def get_answer_prompt(benchmark: str, is_instruct: bool) -> str:
    """Return the prompt template for answer generation."""
    templates = {
        ("math", True): MATH_INSTRUCT,
        ("math", False): MATH_BASE_FEWSHOT,
        ("hotpotqa", True): HOTPOTQA_INSTRUCT,
        ("hotpotqa", False): HOTPOTQA_BASE_FEWSHOT,
    }
    key = (benchmark, is_instruct)
    if key not in templates:
        raise ValueError(f"Unknown benchmark/instruct combo: {key}")
    return templates[key]


def get_question_field(benchmark: str) -> str:
    """Return the field name containing the question/problem text."""
    fields = {"math": "problem", "hotpotqa": "question"}
    if benchmark not in fields:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return fields[benchmark]


def get_confidence_prompt(is_instruct: bool) -> str:
    """Return the confidence elicitation prompt template."""
    return VERBALIZED_CONF_INSTRUCT if is_instruct else VERBALIZED_CONF_BASE
