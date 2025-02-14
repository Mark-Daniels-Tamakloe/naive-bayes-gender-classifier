#!/usr/bin/python
import numpy as np
import hashlib
import string

def stable_hash(text, d):
    """Creates a stable hash function using MD5."""
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % d

def name2features(name):
    """
    Enhanced name feature extraction with gender-specific patterns:
    - Original features (prefix/suffix, n-grams, etc.)
    - Gender-specific ending patterns
    - Enhanced position-aware features
    """
    d = 1024
    v = np.zeros(d)
    name = name.lower()

    vowels = set("aeiou")
    alphabet = string.ascii_lowercase

    # === Original Features ===
    # Prefix Hashing with increased weight for shorter prefixes
    for m in range(4):
        if len(name) > m:
            prefix_sub = f'prefix_{name[:m+1]}'
            v[stable_hash(prefix_sub, d)] += 1.5 if m < 2 else 1

    # Suffix Hashing with increased weight for endings
    for m in range(4):
        if len(name) > m:
            suffix_sub = f'suffix_{name[-(m+1):]}'
            v[stable_hash(suffix_sub, d)] += 2 if m < 2 else 1

    # Bi-gram Encoding with position awareness
    bigram_count = max(1, len(name) - 1)
    for i in range(len(name) - 1):
        bigram = name[i:i+2]
        weight = 1.5 if i == len(name) - 2 else 1.0  # Higher weight for ending
        v[stable_hash(f'bi_{bigram}', d)] += weight / bigram_count

    # Tri-gram Encoding with ending focus
    trigram_count = max(1, len(name) - 2)
    for i in range(len(name) - 2):
        trigram = name[i:i+3]
        weight = 1.5 if i == len(name) - 3 else 1.0  # Higher weight for ending
        v[stable_hash(f'tri_{trigram}', d)] += weight / trigram_count

    # === Gender-Specific Patterns ===
    # Common female name endings
    female_endings = ['a', 'ia', 'na', 'ine', 'elle', 'ie', 'ey', 'anne', 'ette', 'lyn']
    for ending in female_endings:
        if name.endswith(ending):
            v[stable_hash(f'fem_end_{ending}', d)] += 1.5

    # Common male name endings
    male_endings = ['n', 'o', 'er', 'on', 'an', 'ck', 'ton', 'son', 'ert', 'vin']
    for ending in male_endings:
        if name.endswith(ending):
            v[stable_hash(f'male_end_{ending}', d)] += 1.5

    # Character frequency with position weighting
    for i, char in enumerate(name):
        if char in alphabet:
            weight = 2.0 if i == len(name) - 1 else 1.0  # Double weight for last letter
            v[stable_hash(f'char_{char}', d)] += np.log1p(weight)

    # Vowel Features
    num_vowels = sum(1 for ch in name if ch in vowels)
    v[-3] = num_vowels / len(name) if len(name) > 0 else 0

    # Enhanced ending features
    v[-2] = 2.0 if name and name[-1] in vowels else 0  # Doubled weight for vowel ending

    # Length feature
    max_length = 15
    v[-1] = min(len(name) / max_length, 1)

    # Normalize
    norm = np.linalg.norm(v)
    if norm > 0:
        v /= norm

    return v