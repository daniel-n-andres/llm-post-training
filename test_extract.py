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

    total_cases = (len(DIRECT_ANSWER_CASES) + len(CHAIN_OF_THOUGHT_CASES)
                   + len(FIRST_VS_LAST_CASES))
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


if __name__ == "__main__":
    raise SystemExit(1 if run_checks() else 0)
