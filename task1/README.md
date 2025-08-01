# Task 1
## Overview
This repository contains materials for Task 1 of the assignment, focused on prompt evaluation using Promptfoo. The structure includes:

- **`prompts` folder**: Three system prompts for evaluation.
- **`test_cases.json`**: Test cases for Promptfoo.
- **`generate_config.py`**: Script to generate `promptfooconfig.yaml` from `test_cases.json`.

Running `generate_config.py` creates a new `promptfooconfig.yaml`. Evaluation results are saved to `results.json`. Re-running the setup will overwrite both files.

## Models Used
Three models are referenced:

- **`ollama:phi:latest`**: Main evaluation model ([Ollama Phi model](https://ollama.com/library/phi)).
- **`ollama:mistral:latest`**: Larger model (7B parameters).
- **`ollama:llama3:70b`**: Used for specific scenarios.

For conversational, face-to-face evaluations, `mistral:latest` was excluded due to performance limitations. In these cases, `llama3:70b` was used instead.

## Results
From the pure tests the best model+prompt combination was mistral with the friendly prompt passing all tests.
Overall mistrall outperforms phi on all prompts, with phis best result beeing on the friendly prompt matching the performance of mistral on other prompts.

It should be noted that even with llama3 the evaluation if a response looks like a face-to-face conversation is flaky and stays undetected at times.
Further static tests could improve on this issue.


