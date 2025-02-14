#!/usr/bin/python
import numpy as np
import random

def name2features(name):
    """
    Converts a name into a numerical feature vector.
    The feature vector consists of:
    - Prefixes (up to 3 letters)
    - Suffixes (up to 3 letters)
    - Letter bi-grams and tri-grams
    - Vowel-related features
    - Normalized TF-IDF style weighting

    Returns:
    - A vector of length d representing the extracted features.
    """

    d = 512  # Increased feature space to reduce collisions
    v = np.zeros(d)
    name = name.lower()

    vowels = set("aeiou")

    # Hash Prefixes (up to 3 letters)
    prefix_max = 3
    for m in range(prefix_max):
        prefix_string = 'prefix' + name[:m+1]  # First m+1 characters
        prefix_index = hash(prefix_string) % d
        v[prefix_index] += 1  # Count occurrences instead of binary

    # Hash Suffixes (up to 3 letters)
    suffix_max = 3
    for m in range(suffix_max):
        suffix_string = 'suffix' + name[-(m+1):]  # Last m+1 characters
        suffix_index = hash(suffix_string) % d
        v[suffix_index] += 1  # Count occurrences instead of binary

    # Hash Letter Bi-grams (2-letter sequences)
    for i in range(len(name) - 1):
        bigram = f"bi_{name[i]}{name[i+1]}"
        bigram_index = hash(bigram) % d
        v[bigram_index] += 1  # Count occurrences

    # Hash Letter Tri-grams (3-letter sequences)
    for i in range(len(name) - 2):
        trigram = f"tri_{name[i]}{name[i+1]}{name[i+2]}"
        trigram_index = hash(trigram) % d
        v[trigram_index] += 1  # Count occurrences

    # Vowel Ratio Feature (Normalized)
    vowel_count = sum(1 for char in name if char in vowels)
    vowel_ratio = vowel_count / len(name) if len(name) > 0 else 0
    v[-1] = vowel_ratio  # Store normalized vowel ratio

    # Binary Feature: Does the name end in a vowel?
    v[-2] = 1 if name[-1] in vowels else 0

    # Normalize using TF-IDF style weighting
    v = v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v

    return v
