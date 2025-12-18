import random
import string

ALPHABET = string.ascii_lowercase

def rand_word(k=4):
    return "".join(random.choices(ALPHABET, k=k))

def base_block(n=4):
    return [rand_word() for _ in range(n)]

def pattern_A():
    b = base_block()
    return " ".join(b + b)

def pattern_B():
    b = base_block()
    return " ".join(b + b[::-1])

def pattern_C():
    b = base_block()
    return " ".join(b + b[1:] + b[:1])

def pattern_D():
    b = base_block()
    anchor = "zzzz"
    return " ".join([anchor] + b + [anchor] + b)

def pattern_E():
    b = base_block()
    return " ".join(b + ["|"] + b)

PATTERNS = [pattern_A, pattern_B, pattern_C, pattern_D, pattern_E]

def generate_corpus(
    n_lines=100_000,
    pattern_weights=(0.25, 0.25, 0.2, 0.15, 0.15),
    seed=42
):
    random.seed(seed)
    lines = []

    for _ in range(n_lines):
        p = random.choices(PATTERNS, weights=pattern_weights, k=1)[0]
        lines.append(p())

    return "\n".join(lines)

# ======= 生成数据 =======
corpus = generate_corpus(
    n_lines=200_000,   # 可改成 1e4 / 1e6
)

with open("synthetic_attention_corpus.txt", "w") as f:
    f.write(corpus)

print("✓ Synthetic multi-pattern corpus generated")
