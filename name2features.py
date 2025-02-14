#!/usr/bin/python
import numpy as np
import hashlib
import string

def stable_hash(text, d):
    """Creates a stable hash function using MD5."""
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % d

def name2features(name):
    """
    Converts a name into a feature vector using multiple techniques:
    - Prefix and Suffix hashing (4-character sequences)
    - Bi-gram and Tri-gram character encoding (normalized)
    - Character frequency encoding with log scaling
    - Name length as a normalized feature
    - Vowel ratio as a feature
    - Binary vowel ending indicator
    - Enhanced weighting for rare character pairs
    """

    d = 1024  # Original feature space
    v = np.zeros(d)
    name = name.lower()

    vowels = set("aeiou")
    alphabet = string.ascii_lowercase

    # Prefix Hashing (Up to 4 characters) - Exactly as original
    for m in range(4):
        if len(name) > m:
            prefix_sub = f'prefix_{name[:m+1]}'
            v[stable_hash(prefix_sub, d)] += 1

    # Suffix Hashing (Up to 4 characters) - Exactly as original
    for m in range(4):
        if len(name) > m:
            suffix_sub = f'suffix_{name[-(m+1):]}'
            v[stable_hash(suffix_sub, d)] += 1

    # Bi-gram Encoding with rare pair bonus
    bigram_count = max(1, len(name) - 1)
    for i in range(len(name) - 1):
        bigram = name[i:i+1]
        v[stable_hash(f'bi_{bigram}', d)] += 1 / bigram_count
        # Small bonus for rare consonant pairs
        if (name[i] not in vowels and name[i+1] not in vowels):
            v[stable_hash(f'bi_{bigram}', d)] += 0.1

    # Tri-gram Encoding (Normalized) - Exactly as original
    trigram_count = max(1, len(name) - 2)
    for i in range(len(name) - 2):
        trigram = f'tri_{name[i]}{name[i+1]}{name[i+2]}'
        v[stable_hash(trigram, d)] += 1 / trigram_count

    # Log-scaled Character Frequency Encoding - Exactly as original
    char_count = {char: np.log(1 + name.count(char)) for char in alphabet if char in name}
    for char, freq in char_count.items():
        v[stable_hash(f'char_{char}', d)] += freq

    # Vowel Ratio Feature - Exactly as original
    num_vowels = sum(1 for ch in name if ch in vowels)
    v[-3] = num_vowels / len(name) if len(name) > 0 else 0

    # Vowel Ending Binary Feature - Exactly as original
    v[-2] = 1 if name[-1] in vowels else 0

    # Normalized Name Length Feature - Exactly as original
    max_length = 15
    v[-1] = min(len(name) / max_length, 1)

    # Normalize vector - Exactly as original
    v = v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v

    return v