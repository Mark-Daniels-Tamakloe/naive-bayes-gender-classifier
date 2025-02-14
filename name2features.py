#!/usr/bin/python
import numpy as np

def name2features(name):
    """
    Converts a name into a binary feature vector.
    The vector encodes prefix, suffix, letter bigrams, and vowel ratio.
    """

    d = 256  # Increase the number of hashing buckets for better distribution
    v = np.zeros(d)
    name = name.lower()

    # **1. Extended Prefix & Suffix (4 Characters)**
    prefix_max = 4
    for m in range(prefix_max):
        prefix_string = "prefix_" + name[: min(m + 1, len(name))]
        prefix_index = hash(prefix_string) % d
        v[prefix_index] = 1

    suffix_max = 4
    for m in range(suffix_max):
        suffix_string = "suffix_" + name[-min(m + 1, len(name)) :]
        suffix_index = hash(suffix_string) % d
        v[suffix_index] = 1

    # **2. Letter Bi-grams (Captures Phonetic Patterns)**
    for i in range(len(name) - 1):
        bigram = f"bi_{name[i]}{name[i+1]}"
        bigram_index = hash(bigram) % d
        v[bigram_index] = 1

    # **3. Vowel Ratio Feature**
    vowels = set("aeiou")
    vowel_count = sum(1 for char in name if char in vowels)
    vowel_ratio = vowel_count / len(name) if len(name) > 0 else 0

    # Assign a specific feature slot for vowel ratio
    v[-1] = 1 if vowel_ratio > 0.5 else 0  # 1 if > 50% vowels, else 0

    return (v > 0).astype(int)  # Convert to strictly binary values
