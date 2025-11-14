# Autonomous Coding Agent Guidelines

This document contains guidance for any code-generation agent collaborating on this project.

Please also review the repository documentation for additional context:
- [README.md](README.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- Additional references in the `docs/` directory

## Commit Messages

Use the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification for all commit messages.
Example conventional commit: `feat: add dataset summary command`.
Keep commit bodies concise and focused on what changed and why.

Human contributors should review [CONTRIBUTING.md](CONTRIBUTING.md) for repository-wide workflows and expectations.

## Testing

Always use the project's justfile for running tests:

```bash
just test
```

Do NOT run pytest directly. The justfile ensures proper configuration and environment setup.

## Pull Requests

- Keep changes tightly scoped; split work into separate PRs when needed.
- Summarize modifications and corresponding tests in the final hand-off message.
- Reference relevant issues or tickets in the PR description when applicable.

## Project Structure

This project follows a layered architecture:

```
CLI Layer (cli.py)
    ↓
Commands Layer (commands/)
    ↓
Services Layer (services/)
    ↓
Models Layer (models/)
    ↓
External Dependencies (DSPy, MLFlow, HuggingFace Datasets)
```

## Code Patterns

### Imports
- Use top-level, absolute imports only
- NO guarded imports (imports inside functions)
- All imports should be at the module level

### Services
- Use lazy loading and caching for expensive operations
- Store cached data in private attributes (prefixed with `_`)
- Raise `RuntimeError` if dependencies aren't loaded yet
- Follow existing patterns in `services/dataset.py`

### Commands
- Use Pydantic models for request/response validation
- Commands delegate to services (no business logic in commands)
- Follow existing patterns in `commands/dataset/`

### Models
- Use Pydantic for data validation
- Use Pandera for DataFrame schema validation
- Type hints throughout

## Dataset Splits

The dataset has the following splits:
- `train`: Training data (after 80/20 split: ~853 rows)
- `validation`: Validation data (after 80/20 split: ~171 rows)
- `test`: Test data (original test split)
- `all`: All data concatenated

The train/validation split is deterministic (random_state=42) and cached to ensure consistency.

## Development Workflow

1. Make changes to code
2. Run tests: `just test`
3. Verify via CLI commands (e.g., `symptom-diagnosis-explorer dataset summary --split validation`)
4. Check types/linting if needed (see justfile for available commands)
