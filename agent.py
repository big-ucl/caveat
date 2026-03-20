"""LLM-assisted model editor — calls Claude to propose hyperparameter/architecture edits."""

import ast
import json
import re
import textwrap
from pathlib import Path

import anthropic

SYSTEM_PROMPT = """\
You are an ML hyperparameter and architecture tuning assistant.

You will be given:
1. The full source of a Python model file
2. Metrics from the most recent training run
3. The history of all previous runs

Your job is to improve the model by editing the file. You may only modify these six functions:
- get_hyperparameters
- build_model
- get_datamodule
- _Model.__init__
- _Model.forward
- _Model.training_step
- _Model.validation_step

Output ONLY a bare JSON array of edit operations — no prose, no markdown fences, no explanation.

Edit operation types:

1. Replace a top-level function entirely:
   {"op": "replace_function", "function_name": "get_hyperparameters", "new_source": "def get_hyperparameters():\\n    return {...}"}

2. Replace a range of lines (1-indexed, inclusive):
   {"op": "replace_lines", "start": 45, "end": 52, "new_lines": ["        x = self.bn(x)"]}

3. Insert lines after a given line (1-indexed):
   {"op": "insert_after", "after_line": 80, "new_lines": ["        x = F.relu(x)"]}

Rules:
- Output a valid JSON array. No other text.
- Make targeted, incremental improvements based on the metrics trend.
- Do not remove required contract functions (get_hyperparameters, build_model, get_datamodule).
- Preserve the val_loss, val_accuracy, train_loss logging calls.
"""


def syntax_check(source: str) -> str | None:
    """Check Python source for syntax errors.

    Args:
        source: Python source code string.

    Returns:
        Error message string if invalid, None if valid.
    """
    try:
        ast.parse(source)
        return None
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}"


def _replace_function(lines: list[str], fname: str, new_src: str) -> list[str]:
    """Replace a function definition in source lines using AST location.

    Args:
        lines: Source file lines (without newlines).
        fname: Name of function to replace.
        new_src: New function source code.

    Returns:
        Updated lines list.

    Raises:
        ValueError: If the function is not found.
    """
    source = "\n".join(lines)
    tree = ast.parse(source)

    start_line = None
    end_line = None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fname:
            start_line = node.lineno  # 1-indexed
            end_line = node.end_lineno  # 1-indexed inclusive
            break

    if start_line is None:
        raise ValueError(f"Function '{fname}' not found in source")

    # Detect indentation of the original function definition
    original_line = lines[start_line - 1]
    indent = len(original_line) - len(original_line.lstrip())
    indent_str = original_line[: indent]

    # Re-indent new_src to match original indentation
    new_lines = textwrap.dedent(new_src).splitlines()
    if indent > 0:
        new_lines = [indent_str + line if line.strip() else line for line in new_lines]

    return lines[: start_line - 1] + new_lines + lines[end_line:]


def apply_edits(source: str, edits: list[dict]) -> str:
    """Apply a list of edit operations to source code.

    replace_function ops use AST and are applied first (order-independent).
    replace_lines and insert_after ops are applied in forward order after.

    Args:
        source: Original Python source.
        edits: List of edit operation dicts.

    Returns:
        Modified source string.
    """
    lines = source.splitlines()

    # Apply replace_function ops first (AST-based, order-independent)
    for edit in edits:
        if edit["op"] == "replace_function":
            lines = _replace_function(lines, edit["function_name"], edit["new_source"])

    # Apply line-based ops in forward order
    line_edits = [e for e in edits if e["op"] in ("replace_lines", "insert_after")]
    offset = 0  # cumulative line offset from previous insertions/deletions

    for edit in line_edits:
        if edit["op"] == "replace_lines":
            start = edit["start"] - 1 + offset  # convert to 0-indexed
            end = edit["end"] + offset  # exclusive end
            new = edit["new_lines"]
            old_len = end - start
            lines = lines[:start] + new + lines[end:]
            offset += len(new) - old_len

        elif edit["op"] == "insert_after":
            after = edit["after_line"] - 1 + offset  # 0-indexed position
            new = edit["new_lines"]
            lines = lines[: after + 1] + new + lines[after + 1 :]
            offset += len(new)

    return "\n".join(lines)


def get_edits(model_path: str, metrics: dict, history: list[dict]) -> list[dict]:
    """Ask Claude to propose edits to improve the model.

    Args:
        model_path: Path to the user model file.
        metrics: Metrics dict from the most recent run.
        history: List of all previous run metrics dicts.

    Returns:
        List of edit operation dicts.
    """
    source = Path(model_path).read_text()

    user_message = f"""## Model source

```python
{source}
```

## Latest run metrics

```json
{json.dumps(metrics, indent=2)}
```

## Run history (all runs so far)

```json
{json.dumps(history, indent=2)}
```

Propose edits to improve validation loss and/or accuracy. Output only the JSON array."""

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)


def edit_model(model_path: str, metrics: dict, history: list[dict]) -> str:
    """Get LLM edits, apply them, syntax-check, and write back to file.

    Args:
        model_path: Path to the user model file.
        metrics: Metrics dict from the most recent run.
        history: List of all previous run metrics dicts.

    Returns:
        The updated source code string.

    Raises:
        ValueError: If the edited source fails syntax check.
    """
    edits = get_edits(model_path, metrics, history)
    source = Path(model_path).read_text()
    new_source = apply_edits(source, edits)

    error = syntax_check(new_source)
    if error:
        raise ValueError(f"Edited source has syntax error: {error}")

    Path(model_path).write_text(new_source)
    return new_source
