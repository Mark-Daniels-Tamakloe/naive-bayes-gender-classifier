#!/usr/bin/python
import numpy as np
import hashlib
import string

def stable_hash(text, d):
    """Creates a stable hash function using MD5."""
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % d

def name2features(name):
    """
    Converts a name into a feature vector using carefully optimized techniques:
    - Original prefix/suffix hashing with enhanced weighting
    - Separate normalization for bi-grams and tri-grams
    - Optimized character frequency encoding
    - Core statistical features from original
    """
    d = 1024  # Original dimension size
    v = np.zeros(d)
    name = name.lower()

    vowels = set("aeiou")
    alphabet = string.ascii_lowercase

    # === Prefix/Suffix Hashing (Original Method) ===
    # Keeping original exact implementation as it worked well
    for m in range(4):
        if len(name) > m:
            prefix_sub = f'prefix_{name[:m+1]}'
            v[stable_hash(prefix_sub, d)] += 1
            
            suffix_sub = f'suffix_{name[-(m+1):]}'
            v[stable_hash(suffix_sub, d)] += 1

    # === Bi-gram Encoding with Original Normalization ===
    bigram_count = max(1, len(name) - 1)
    for i in range(len(name) - 1):
        bigram = f'bi_{name[i]}{name[i+1]}'
        v[stable_hash(bigram, d)] += 1 / bigram_count

    # === Tri-gram Encoding with Original Normalization ===
    trigram_count = max(1, len(name) - 2)
    for i in range(len(name) - 2):
        trigram = f'tri_{name[i]}{name[i+1]}{name[i+2]}'
        v[stable_hash(trigram, d)] += 1 / trigram_count

    # === Character Frequency (Enhanced) ===
    # Using log scaling with slightly higher weight for first/last chars
    for i, char in enumerate(name):
        if char in alphabet:
            weight = 1.2 if (i == 0 or i == len(name) - 1) else 1.0
            v[stable_hash(f'char_{char}', d)] += np.log1p(weight)

    # === Original Statistical Features ===
    # Keeping these exactly as they were since they worked well
    num_vowels = sum(1 for ch in name if ch in vowels)
    v[-3] = num_vowels / len(name) if len(name) > 0 else 0

    # Vowel ending feature
    v[-2] = 1 if name and name[-1] in vowels else 0

    # Length feature
    max_length = 15
    v[-1] = min(len(name) / max_length, 1)

    # === Original Normalization ===
    # Keep exact original normalization
    norm = np.linalg.norm(v)
    if norm > 0:
        v /= norm

    return v