#!/usr/bin/python
import numpy as np
import hashlib
import string

def stable_hash(text, d):
    """Creates a stable hash function using MD5."""
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % d

def name2features(name):
    """
    Optimized name feature extraction focusing on most predictive features:
    - Enhanced prefix/suffix processing
    - Carefully weighted n-grams
    - Optimized character frequency encoding
    - Key statistical features
    """
    d = 1024  # Keep original dimension - it was working well
    v = np.zeros(d)
    name = name.lower().strip()
    
    if not name:
        return v
        
    vowels = set("aeiou")
    alphabet = string.ascii_lowercase
    
    # === Prefix and Suffix Features (Enhanced) ===
    # Weight prefixes/suffixes by length (shorter ones get higher weight)
    for m in range(1, 5):
        if len(name) >= m:
            prefix = name[:m]
            suffix = name[-m:]
            # Higher weight for shorter n-grams as they're more common
            weight = 1.0 / m
            v[stable_hash(f'pfx_{prefix}', d)] += weight
            v[stable_hash(f'sfx_{suffix}', d)] += weight
    
    # === Optimized N-gram Features ===
    # Bi-grams (with position awareness for start/end)
    if len(name) >= 2:
        for i in range(len(name) - 1):
            bigram = name[i:i+2]
            if i == 0:  # Start bigram
                v[stable_hash(f'start_bi_{bigram}', d)] += 1.5
            elif i == len(name) - 2:  # End bigram
                v[stable_hash(f'end_bi_{bigram}', d)] += 1.5
            else:  # Middle bigrams (lower weight)
                v[stable_hash(f'bi_{bigram}', d)] += 0.8
    
    # Tri-grams (focused on middle of name)
    if len(name) >= 3:
        for i in range(len(name) - 2):
            trigram = name[i:i+3]
            v[stable_hash(f'tri_{trigram}', d)] += 0.6  # Lower weight for trigrams
    
    # === Character Frequency Features ===
    # Log-scaled character frequency with position bonus
    char_counts = {}
    for i, char in enumerate(name):
        if char in alphabet:
            # Add position-based weighting
            position_weight = 1.2 if i == 0 or i == len(name)-1 else 1.0
            char_counts[char] = char_counts.get(char, 0) + position_weight
    
    for char, count in char_counts.items():
        v[stable_hash(f'char_{char}', d)] += np.log1p(count)
    
    # === Core Statistical Features ===
    # Vowel ratio (keeping this from original as it worked well)
    num_vowels = sum(1 for ch in name if ch in vowels)
    v[-3] = num_vowels / len(name)
    
    # Vowel ending (keeping this from original)
    v[-2] = 1.0 if name[-1] in vowels else 0.0
    
    # Length feature (simplified)
    v[-1] = min(len(name) / 15.0, 1.0)  # Normalize to [0,1]
    
    # === Final Normalization ===
    # L2 normalization
    norm = np.linalg.norm(v)
    if norm > 0:
        v /= norm
    
    return v