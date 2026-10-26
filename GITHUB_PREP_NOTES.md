# GitHub preparation notes

This package is a cleaned copy of the uploaded project, not a live change to your GitHub account.

Changes made for this package:
- Added a project README and `requirements.txt`.
- Added `.gitignore` for Python caches, local databases, logs, environments, and raw-data path.
- Excluded local SQLite databases, generated logs, `__pycache__`/`.pyc`, and `data/raw`.
- Fixed the obvious syntax typo `print(z` in `evaluate_ocsvm.py`.

Before publishing:
- Check redistribution terms for HDFS data and any included datasets.
- Re-run evaluation only after verifying labels align row-for-row with the test logs.
- Verify the API prototype and dependency versions in a fresh virtual environment.
- Do not publish confidential internship logs or internal information.
