"""
Tests for answer extraction — the quiet failure point.

`extract_answer(use_last_number=True)` is used for chain-of-thought, where the answer is
the final number after the working. Get that wrong and the CoT condition
silently scores an intermediate step, which looks like CoT hurting performance
when really the parser is broken. That is exactly the ambiguity these tests
exist to remove.

    python -m pytest test_extract.py -q      # or just: python test_extract.py
"""

from model import extract_answer

# (model output, expected answer) — the answer is the FIRST number present.
DIRECT_ANSWER_CASES = [
    ("750", 750),
    (" 750", 750),
    ("750\n", 750),
    ("The answer is 750", 750),
    ("750 is the result", 750),
    ("1,234", 1234),
    ("-42", -42),
    ("", None),
    ("I cannot help with that", None),
]

# What CoT output actually looks like: the final number is the answer,
# every earlier number is working.
CHAIN_OF_THOUGHT_CASES = [
    ("15 * 50 = 15 * 5 * 10 = 75 * 10 = 750", 750),
    ("First, 15 * 5 = 75.\nThen 75 * 10 = 750.\n750", 750),
    ("Step 1: 20 + 30 = 50\nStep 2: 50 + 7 = 57\nFinal answer: 57", 57),
    ("Let me think. 9 x 9 = 81. So the answer is 81.", 81),
]

# (model output, expected first, expected last) — cases where first-vs-last
# matters, so the flag is doing real work.
FIRST_VS_LAST_CASES = [
    ("Question 48: 15 * 50. The answer is 750.", 48, 750),
    ("Item 3 of 50: 12 + 5 = 17", 3, 17),
]

# Shapes taken from real Qwen traces. These must keep working — they are the
# common case and the old parser handled them, so they are regression cover,
# not evidence of a fix.
REAL_TRACE_CASES = [
    ("To solve this multiplication problem:\n"
     "1) 46 * 90 = 4140\n"
     "2) 92 / 5 = 18.4\n"
     "Final answer: 4232", 4232),
    ("Step 1: The first number is 7.\n"
     "Step 2: 7 * 3 = 21\n"
     "\n"
     "Final answer:\n21", 21),
    ("To solve \\( 81 \\times 57 \\):\n"
     "Step 1: \\( 81 \\times 7 = 567 \\)\n"
     "Step 2: \\( 81 \\times 50 = 4050 \\)\n"
     "So the product is 4617.", 4617),
    ("15 * 50\n\\boxed{750}", 750),
    ("3 * 8 = 24\n\nFinal answer: 24\n\n  \n", 24),
    ("Working it through.\nThe total is 12,345", 12345),
]

# Verbatim shapes from the base run — these are the ones that matter, and
# they are here because an invented test set led to a parser change that lost
# 5 items per 200. This model restates the problem inside its announcement, so
# the answer is the LAST number, not the first one after "Final answer:".
ANNOUNCEMENT_CASES = [
    ("Step 2: Multiply these two numbers together.\n"
     "    - 2 multiplied by 6 equals 12.\n"
     "Final answer: The result of 2 * 6 is 12.", 12),
    ("    - 3 multiplied by 2 equals 6.\n"
     "Final answer: The result of 3 * 2 is 6.", 6),
    ("    - 8 * 2 = 16\n"
     "Final answer: The result of 8 multiplied by 2 is 16.", 16),
    # Model is wrong, parser is right — the parse must report what the model
    # actually concluded, not hunt the text for the correct answer.
    ("Step 3: Add the results: 8 + 400 = 408\n"
     "Therefore, the final answer is 408.", 408),
]

# Truncated working with a decimal as the last number. No answer was stated,
# so there is no right value to assert — the requirement is narrower: do not
# manufacture an integer out of half a decimal, which is how the old parser
# turned "18.4" into a confident 4.
MUST_NOT_RETURN = [
    ("46 * 2 = 92\nThen 92 / 5 = 18.4", 4),
    ("Half of it is 9.5", 5),
]

# Truncated mid-working: no answer was ever stated, so no parser could recover
# one. The contract here is only that extraction does not raise and does not
# invent an integer out of half a decimal.
TRUNCATED_CASES = [
    "2) 92 / 5 = 18.4\n\n3) Finally, add",
    "- 30 * 50 = 1500 (because 3 times 5 is 15)\n    - 30 * 7 = 210 (because 3",
]


def run_checks() -> int:
    """Print every mismatch and return how many cases failed."""
    failure_count = 0
    for model_output, expected_answer in DIRECT_ANSWER_CASES:
        parsed_answer = extract_answer(model_output)
        if parsed_answer != expected_answer:
            failure_count += 1
            print(f"  DIRECT  {model_output!r}: "
                  f"got {parsed_answer}, want {expected_answer}")

    for model_output, expected_answer in CHAIN_OF_THOUGHT_CASES:
        parsed_answer = extract_answer(model_output, use_last_number=True)
        if parsed_answer != expected_answer:
            failure_count += 1
            print(f"  COT     {model_output!r}: "
                  f"got {parsed_answer}, want {expected_answer}")

    for model_output, expected_first, expected_last in FIRST_VS_LAST_CASES:
        parsed_first = extract_answer(model_output)
        parsed_last = extract_answer(model_output, use_last_number=True)
        if parsed_first != expected_first or parsed_last != expected_last:
            failure_count += 1
            print(f"  CONTRAST {model_output!r}: "
                  f"first {parsed_first} (want {expected_first}), "
                  f"last {parsed_last} (want {expected_last})")

    for model_output, expected_answer in REAL_TRACE_CASES:
        parsed_answer = extract_answer(model_output, use_last_number=True)
        if parsed_answer != expected_answer:
            failure_count += 1
            print(f"  TRACE   {model_output!r}: "
                  f"got {parsed_answer}, want {expected_answer}")

    for model_output, expected_answer in ANNOUNCEMENT_CASES:
        parsed_answer = extract_answer(model_output, use_last_number=True)
        if parsed_answer != expected_answer:
            failure_count += 1
            print(f"  ANNOUNCE {model_output!r}: "
                  f"got {parsed_answer}, want {expected_answer}")

    for model_output, forbidden_answer in MUST_NOT_RETURN:
        parsed_answer = extract_answer(model_output, use_last_number=True)
        if parsed_answer == forbidden_answer:
            failure_count += 1
            print(f"  DECIMAL {model_output!r}: returned {parsed_answer}, "
                  "which is a fragment of a decimal")

    for model_output in TRUNCATED_CASES:
        try:
            extract_answer(model_output, use_last_number=True)
        except Exception as error:
            failure_count += 1
            print(f"  TRUNC   {model_output!r}: raised {error!r}")

    total_cases = (len(DIRECT_ANSWER_CASES) + len(CHAIN_OF_THOUGHT_CASES)
                   + len(FIRST_VS_LAST_CASES) + len(REAL_TRACE_CASES)
                   + len(ANNOUNCEMENT_CASES) + len(MUST_NOT_RETURN)
                   + len(TRUNCATED_CASES))
    print(f"{total_cases - failure_count}/{total_cases} passed"
          if failure_count else f"all {total_cases} passed")
    return failure_count


# pytest entry points
def test_extracts_direct_answers():
    for model_output, expected_answer in DIRECT_ANSWER_CASES:
        assert extract_answer(model_output) == expected_answer, model_output


def test_chain_of_thought_takes_last_number():
    for model_output, expected_answer in CHAIN_OF_THOUGHT_CASES:
        assert extract_answer(model_output, use_last_number=True) == expected_answer, model_output


def test_first_and_last_extract_differently():
    for model_output, expected_first, expected_last in FIRST_VS_LAST_CASES:
        assert extract_answer(model_output) == expected_first, model_output
        assert extract_answer(model_output, use_last_number=True) == expected_last, model_output


def test_real_traces():
    for model_output, expected_answer in REAL_TRACE_CASES:
        assert extract_answer(model_output, use_last_number=True) == expected_answer, model_output


def test_announcement_does_not_beat_the_last_number():
    """Real traces restate the problem inside the announcement, so reading
    forward from "Final answer:" grabs an operand. Measured cost of getting
    this wrong: 5 items per 200."""
    for model_output, expected_answer in ANNOUNCEMENT_CASES:
        assert extract_answer(model_output, use_last_number=True) == expected_answer, model_output


def test_decimals_are_not_split():
    for model_output, forbidden_answer in MUST_NOT_RETURN:
        assert extract_answer(model_output, use_last_number=True) != forbidden_answer, model_output


def test_truncated_traces_do_not_raise():
    for model_output in TRUNCATED_CASES:
        extract_answer(model_output, use_last_number=True)


if __name__ == "__main__":
    raise SystemExit(1 if run_checks() else 0)
