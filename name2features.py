#!/usr/bin/python
import numpy as np
import random
import string

def name2features(name):
    """
    Converts a name into a feature vector using multiple techniques:
    - Prefix and Suffix hashing (3 characters max)
    - Bi-gram and Tri-gram character encoding
    - Character frequency encoding (relative frequency)
    - Name length as a feature
    - Binary vowel ending indicator
    - TF-IDF normalization for weight balancing
    """

    d = 768  # Increased feature space for better hashing distribution
    v = np.zeros(d)
    name = name.lower()

    vowels = set("aeiou")
    alphabet = string.ascii_lowercase

    # Prefix Hashing (Up to 3 characters)
    for m in range(3):
        if len(name) > m:
            prefix_string = f'prefix{name[:m+1]}'
            prefix_index = hash(prefix_string) % d
            v[prefix_index] += 1

    # Suffix Hashing (Up to 3 characters)
    for m in range(3):
        if len(name) > m:
            suffix_string = f'suffix{name[-(m+1):]}'
            suffix_index = hash(suffix_string) % d
            v[suffix_index] += 1

    # Bi-gram and Tri-gram Encoding
    for i in range(len(name) - 1):
        bigram = f'bi_{name[i]}{name[i+1]}'
        bigram_index = hash(bigram) % d
        v[bigram_index] += 1

    for i in range(len(name) - 2):
        trigram = f'tri_{name[i]}{name[i+1]}{name[i+2]}'
        trigram_index = hash(trigram) % d
        v[trigram_index] += 1

    # Character Frequency Encoding
    char_count = {char: name.count(char) / len(name) for char in alphabet if char in name}
    for char, freq in char_count.items():
        char_index = hash(f'char_{char}') % d
        v[char_index] += freq  # Weighted frequency

    # Vowel Ending Binary Feature
    v[-1] = 1 if name[-1] in vowels else 0

    # Name Length Feature (Scaled)
    v[-2] = min(len(name) / 10, 1)  # Normalized between 0-1

    # Normalize using TF-IDF style weighting
    v = v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v

    return v
