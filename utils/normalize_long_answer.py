import re


def normalize_long_answer(long_answer):
    parts = re.split(r"Therefore[, ]*the answer is ", long_answer, flags=re.IGNORECASE)
    return parts[-1].strip().rstrip(".") if len(parts) > 1 else ""