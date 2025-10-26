# Claude Code Development Guidelines

This document contains guidance for Claude Code when working on this project.

## Testing

Always use the project's justfile for running tests:

```bash
just test
```

Do NOT run pytest directly. The justfile ensures proper configuration and environment setup.

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
