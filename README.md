# Transformer-based Reversal and Question Classification Models

**Author:** Soli Ateefa
---

## Overview

This project explores two Natural Language Processing (NLP) tasks using transformer-based models implemented with Keras:

1. **Sequence Reversal**
   A model that learns to reverse sequences of tokens, trained on synthetic reversal datasets.

2. **Question Classification**
   A model that classifies questions into categories (e.g., WHO, WHAT, WHERE, etc.) using real-world question data.

Both tasks are implemented from scratch using custom preprocessing, architecture definition, training scripts, and evaluation methods.

---

## File Structure

```bash
assignment7/
│
├── reverse_base.py              # Base implementation for reverse sequence model
├── reverse_common.py            # Shared utilities for reverse task
├── reverse_extra.py             # Optional improvements for reverse model
├── reverse_solution.py          # Final solution for reverse task
├── questions_base.py            # Base implementation for question classification
├── questions_common.py          # Shared utilities for question classification
├── questions_extra.py           # Optional improvements for question model
├── questions_solution.py        # Final solution for classification task
│
├── transformer.keras            # Saved Transformer model file
│
├── reverse_dataset/             # Folder with train/val/test data for reversal
│   ├── reverse_train.txt
│   ├── reverse_validation.txt
│   └── reverse_test.txt
│
├── questions_dataset/           # Folder with question classification data
│   ├── questions_train.txt
│   ├── questions_train_labels.txt
│   ├── questions_validation.txt
│   ├── questions_validation_labels.txt
│   ├── questions_test.txt
│   └── questions_test_labels.txt
│
├── answers.pdf                  # Write-up / report with answers and explanations
└── __pycache__/                 # Auto-generated cache (can be ignored)
```

---

## How to Run

### Requirements

* Python 3.8+
* TensorFlow / Keras (latest version)
* NumPy

You can install the required libraries using pip:

```bash
pip install tensorflow numpy
```

### ▶️ Execution

Navigate to the project directory and run the scripts using Python 3.

**Reversal Task:**

```bash
python3 reverse_solution.py
```

**Question Classification Task:**

```bash
python3 questions_solution.py
```

Both scripts will train the respective models and evaluate them on the test sets. Output will be printed to the terminal.

---

## Notes

* **No compilation needed**: Python scripts run directly.
* The project **does not require ACS Omega** to run — it works locally on any machine with Python 3.8+ and TensorFlow installed.
* `.keras` file stores the trained transformer model for optional reuse or evaluation.

---

## Author's Remarks

This project deepened my understanding of transformer architectures, token embeddings, and classification pipelines. The hands-on practice with both synthetic and real NLP tasks offered valuable insights into preprocessing, architecture tuning, and performance evaluation.

---
