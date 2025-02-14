#!/usr/bin/python
import numpy as np
import hashlib
import string

def stable_hash(text, d):
    """Creates a stable hash function using MD5."""
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % d

def name2features(name):
    """
    Converts a name into a feature vector using:
    - Prefix and Suffix hashing (4-character sequences)
    - Bi-gram, Tri-gram, and Quad-gram encoding (normalized)
    - Character frequency encoding with log scaling
    - Name length as a normalized feature
    - Vowel ratio and rare character indicator
    - First letter encoding (helps capture gender correlations)
    - Middle character encoding for long names
    """

    d = 2304  # Slightly increased feature space
    v = np.zeros(d)
    name = name.lower()

    vowels = set("aeiou")
    rare_chars = set("xqz")
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

    # Quad-gram Encoding (More robust name representation)
    quadgram_count = max(1, len(name) - 3)
    for i in range(len(name) - 3):
        quadgram = f'quad_{name[i]}{name[i+1]}{name[i+2]}{name[i+3]}'
        v[stable_hash(quadgram, d)] += 1 / quadgram_count

    # Log-scaled Character Frequency Encoding
    char_count = {char: np.log(1 + name.count(char)) for char in alphabet if char in name}
    for char, freq in char_count.items():
        v[stable_hash(f'char_{char}', d)] += freq

    # Vowel Ratio Feature
    num_vowels = sum(1 for ch in name if ch in vowels)
    v[-4] = num_vowels / len(name) if len(name) > 0 else 0  # Normalize

    # Rare Character Indicator
    v[-3] = any(ch in rare_chars for ch in name)

    # First Letter Encoding
    v[-2] = stable_hash(f'first_{name[0]}', d)

    # Middle Character Encoding (Only for names longer than 6)
    if len(name) > 6:
        mid_index = len(name) // 2
        v[-1] = stable_hash(f'mid_{name[mid_index]}', d)

    # Normalize vector (Min-Max Scaling for stability)
    v_min, v_max = v.min(), v.max()
    v = (v - v_min) / (v_max - v_min) if v_max > v_min else v

    return v
