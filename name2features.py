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
    - Prefix and Suffix hashing (3 characters max)
    - Bi-gram and Tri-gram character encoding
    - Character frequency encoding (log-scaled)
    - Name length as a log-scaled feature
    - Binary vowel ending indicator
    """

    d = 768  # Expanded feature space
    v = np.zeros(d)
    name = name.lower()

    vowels = set("aeiou")
    alphabet = string.ascii_lowercase

    # Prefix Hashing (Bigrams for robustness)
    for m in range(3):
        if len(name) > m:
            prefix_bigram = f'prefix_{name[:m+1]}'
            v[stable_hash(prefix_bigram, d)] += 1

    # Suffix Hashing (Bigrams for robustness)
    for m in range(3):
        if len(name) > m:
            suffix_bigram = f'suffix_{name[-(m+1):]}'
            v[stable_hash(suffix_bigram, d)] += 1

    # Bi-gram Encoding
    for i in range(len(name) - 1):
        bigram = f'bi_{name[i]}{name[i+1]}'
        v[stable_hash(bigram, d)] += 1

    # Tri-gram Encoding
    for i in range(len(name) - 2):
        trigram = f'tri_{name[i]}{name[i+1]}{name[i+2]}'
        v[stable_hash(trigram, d)] += 1

    # Log-scaled Character Frequency Encoding
    char_count = {char: np.log(1 + name.count(char)) for char in alphabet if char in name}
    for char, freq in char_count.items():
        v[stable_hash(f'char_{char}', d)] += freq

    # Vowel Ending Binary Feature
    v[-1] = 1 if name[-1] in vowels else 0

    # Log-scaled Name Length Feature
    v[-2] = np.log(1 + len(name)) / np.log(20)  # Normalized between 0-1 (assuming max length ~20)

    # Normalize vector (TF-IDF style)
    v = v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v

    return v
