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
    """

    d = 1024  # Expanded feature space (increased from 1024)
    v = np.zeros(d)
    name = name.lower()

    vowels = set("aeiou")
    alphabet = string.ascii_lowercase

    # Prefix Hashing (Up to 4 characters)
    for m in range(4):
        if len(name) > m:
            prefix_sub = f'prefix_{name[:m+1]}'
            v[stable_hash(prefix_sub, d)] += 1

    # Suffix Hashing (Up to 4 characters)
    for m in range(4):
        if len(name) > m:
            suffix_sub = f'suffix_{name[-(m+1):]}'
            v[stable_hash(suffix_sub, d)] += 1

    # Bi-gram Encoding (Normalized)
    bigram_count = max(1, len(name) - 1)  # Avoid division by zero
    for i in range(len(name) - 1):
        bigram = f'bi_{name[i]}{name[i+1]}'
        v[stable_hash(bigram, d)] += 1 / bigram_count  # Normalize by total bi-grams

    # Tri-gram Encoding (Normalized)
    trigram_count = max(1, len(name) - 2)  # Avoid division by zero
    for i in range(len(name) - 2):
        trigram = f'tri_{name[i]}{name[i+1]}{name[i+2]}'
        v[stable_hash(trigram, d)] += 1 / trigram_count  # Normalize by total tri-grams

    # Log-scaled Character Frequency Encoding
    char_count = {char: np.log(1 + name.count(char)) for char in alphabet if char in name}
    for char, freq in char_count.items():
        v[stable_hash(f'char_{char}', d)] += freq

    # Vowel Ratio Feature
    num_vowels = sum(1 for ch in name if ch in vowels)
    v[-3] = num_vowels / len(name) if len(name) > 0 else 0  # Normalize

    # Vowel Ending Binary Feature
    v[-2] = 1 if name[-1] in vowels else 0

    # Normalized Name Length Feature
    max_length = 15  # Assume reasonable max length for names
    v[-1] = min(len(name) / max_length, 1)  # Normalize between 0-1

    # Normalize vector (TF-IDF style)
    v = v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v

    return v
