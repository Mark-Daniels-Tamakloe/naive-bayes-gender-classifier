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
    - Prefix and Suffix hashing (up to 3 characters)
    - Bi-gram and Tri-gram character encoding
    - Character frequency encoding with Min-Max scaling
    - Name length as a normalized feature
    - Binary vowel ending indicator
    - TF-IDF normalization for better weight distribution
    """

    d = 768  # Expanded feature space
    v = np.zeros(d)
    name = name.lower()

    vowels = set("aeiou")
    alphabet = string.ascii_lowercase

    # Prefix Hashing (Limit to 2 characters for better generalization)
    for m in range(2):  
        if len(name) > m:
            prefix_bigram = f'prefix_{name[:m+1]}'
            v[stable_hash(prefix_bigram, d)] += 1

    # Suffix Hashing (Limit to 2 characters for better generalization)
    for m in range(2):  
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

    # Min-Max Scaled Character Frequency Encoding
    char_count = {char: name.count(char) / len(name) for char in alphabet if char in name}
    if char_count:
        min_freq, max_freq = min(char_count.values()), max(char_count.values())
        for char, freq in char_count.items():
            normalized_freq = (freq - min_freq) / (max_freq - min_freq + 1e-6)  # Avoid division by zero
            v[stable_hash(f'char_{char}', d)] += normalized_freq

    # Vowel Ending Binary Feature
    v[-1] = 1 if name[-1] in vowels else 0

    # Normalized Name Length Feature
    max_length = 15  # Assume reasonable max length for names
    v[-2] = min(len(name) / max_length, 1)  # Normalized between 0-1

    # Normalize vector (TF-IDF style)
    v = v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v

    return v
