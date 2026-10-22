# Resonance

Turn support conversations into a ranked list of product issues. Work in progress; full README coming with phase 4.

```bash
pip install -e .
resonance fetch-data
resonance simulate data/bitext.csv --spike payment_issue
resonance analyze simulated-tickets.csv
```
