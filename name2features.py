#!/usr/bin/python
import numpy as np
import hashlib
import string
from collections import Counter

def stable_hash(text, d):
    """Creates a stable hash function using MD5."""
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % d

def name2features(name):
    """
    Enhanced name feature extraction with additional techniques:
    - Improved n-gram coverage (1-4 grams)
    - Position-aware character encoding
    - Consonant patterns
    - Syllable estimation
    - Advanced statistical features
    - Phonetic features
    """
    d = 2048  # Increased feature space for more granular encoding
    v = np.zeros(d)
    name = name.lower().strip()
    
    if not name:  # Handle empty names
        return v
        
    vowels = set("aeiou")
    consonants = set(string.ascii_lowercase) - vowels
    
    # === Character Statistics ===
    # Position-aware character encoding
    for i, char in enumerate(name):
        position_feature = f'pos_{i}_{char}'
        v[stable_hash(position_feature, d)] += 1
        
        # Relative position features (start, middle, end)
        rel_pos = i / len(name)
        if rel_pos < 0.33:
            v[stable_hash(f'start_{char}', d)] += 1
        elif rel_pos > 0.66:
            v[stable_hash(f'end_{char}', d)] += 1
        else:
            v[stable_hash(f'mid_{char}', d)] += 1
    
    # === N-gram Features ===
    # Enhanced n-gram processing (1-4 grams)
    for n in range(1, 5):
        for i in range(len(name) - n + 1):
            ngram = name[i:i+n]
            # Weight n-grams by their length (longer n-grams get higher weight)
            v[stable_hash(f'ng_{ngram}', d)] += n / 4
    
    # === Syllable Features ===
    # Estimate syllables using vowel sequences
    syllable_count = 1
    prev_was_vowel = False
    for char in name:
        if char in vowels:
            if not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = True
        else:
            prev_was_vowel = False
    v[stable_hash('syllables', d)] = np.log1p(syllable_count)
    
    # === Consonant Pattern Features ===
    # Detect consonant clusters
    consonant_pattern = ''.join('C' if c in consonants else 'V' for c in name)
    for i in range(len(consonant_pattern) - 1):
        pattern = consonant_pattern[i:i+2]
        v[stable_hash(f'pattern_{pattern}', d)] += 1
    
    # === Statistical Features ===
    char_counts = Counter(name)
    
    # Character diversity ratio
    diversity = len(char_counts) / len(name)
    v[stable_hash('char_diversity', d)] = diversity
    
    # Consonant-vowel ratio
    consonant_count = sum(1 for c in name if c in consonants)
    vowel_count = sum(1 for c in name if c in vowels)
    cv_ratio = consonant_count / max(1, vowel_count)
    v[stable_hash('cv_ratio', d)] = np.log1p(cv_ratio)
    
    # === Phonetic Features ===
    # Common phonetic endings
    phonetic_endings = ['ing', 'ed', 'er', 'es', 'ion', 'ly', 'ment', 'ness', 'tion']
    for ending in phonetic_endings:
        if name.endswith(ending):
            v[stable_hash(f'phon_{ending}', d)] += 1
    
    # Double letter detection
    for i in range(len(name) - 1):
        if name[i] == name[i + 1]:
            v[stable_hash('double_letter', d)] += 1
    
    # === Length-based Features ===
    # Log-scaled length
    v[stable_hash('length', d)] = np.log1p(len(name))
    
    # Length percentile features (assuming most names are between 2-15 chars)
    length_percentile = min(1.0, len(name) / 15)
    v[stable_hash('length_percentile', d)] = length_percentile
    
    # Normalize the vector
    norm = np.linalg.norm(v)
    if norm > 0:
        v /= norm
    
    return v