# Naive Bayes Gender Classifier

This project implements a Naive Bayes classifier to predict a person's gender based on their first name. Developed as part of a graduate machine learning course at Washington University in St. Louis, the model was built from scratch using Python and NumPy, with an emphasis on understanding probabilistic modeling and feature engineering.

---

## 🔍 Overview

- Predicts gender using character-level features extracted from names.
- Implements Bernoulli Naive Bayes classification with Laplace smoothing.
- Converts probabilistic output into a linear classifier for fast prediction.
- Evaluated using training and test accuracy, as well as user interaction.

---

## 📁 Key Components

- `name2features.py`: Converts names into binary feature vectors.
- `naivebayesPY.py`: Computes prior class probabilities \( P(Y) \).
- `naivebayesPXY.py`: Computes conditional probabilities \( P(X|Y) \).
- `naivebayes.py`: Calculates log posterior for classification.
- `naivebayesCL.py`: Constructs a linear classifier using Bayes estimates.
- `classifyLinear.py`: Applies learned classifier to make predictions.
- `whoareyou.py`: Interactive CLI to test the classifier on new names.

---

## 🧠 Feature Engineering

The classifier uses custom features such as:
- Hashed prefixes and suffixes
- Vowel ratios and character frequency
- Normalized name length
- N-gram indicators (up to trigrams)

---

## 🧪 Performance

| Metric             | Result     |
|--------------------|------------|
| Training Accuracy  | 86%        |
| Testing Accuracy   | 79.7%      |

The model passed all functional correctness tests and nearly met the threshold for bonus accuracy.

---

## 🚀 Usage

To run the interactive classifier:

```bash
python3 whoareyou.py

### 🧪 Example Interaction

```bash
$ python3 whoareyou.py
Who are you> Sophia
Sophia, I am sure you are a nice girl.

Who are you> Michael
Michael, I am sure you are a nice boy.

Who are you> Avery
Avery, I am not quite sure, but I'm guessing you're a boy.

Who are you> exit
Goodbye!



## 🙏 Acknowledgments

This project was developed as part of the Machine Learning course (CSE517A) at Washington University in St. Louis. Original project framework and structure were provided by the course team, including materials adapted from Professor Kilian Q. Weinberger. All implementation, experimentation, and documentation in this repository were completed independently by me.

