# Benchmark report

Each run: the LLM writes one plan for each of 36 test projects. A plan passes only if it passes all six checks.

## 1. Which LLM? (same prompt: v1-structured)

| Model | Plans accepted | Injection tests passed | Seconds per plan |
|---|---|---|---|
| gemma3:4b | 34/36 (94%) | 2/4 | 12.6 |
| llama3.2:3b | 26/36 (72%) | 2/4 | 4.6 |
| qwen2.5:3b | 34/36 (94%) | 2/4 | 2.4 |

## 2. Which prompt? (same model: qwen2.5:3b)

| Prompt | Plans accepted | Injection tests passed |
|---|---|---|
| v0-basic | 0/36 (0%) | 0/4 |
| v1-structured | 34/36 (94%) | 2/4 |
| v2-fewshot | 25/36 (69%) | 1/4 |

## 3. Why plans were rejected

**gemma3:4b / v1-structured**: other invalid plan (2)

**llama3.2:3b / v1-structured**: model name not allowed (8), more than 3 models (2)

**qwen2.5:3b / v0-basic**: search method not allowed (36)

**qwen2.5:3b / v1-structured**: more than 3 models (2)

**qwen2.5:3b / v2-fewshot**: model name not allowed (9), more than 3 models (2)
