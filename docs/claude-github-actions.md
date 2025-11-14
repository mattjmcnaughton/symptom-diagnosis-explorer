# Claude Code GitHub Actions Integration

This document describes the Claude Code GitHub Actions integration configured for this repository.

## Claude Code Setup

### Prerequisites

Before using the Claude Code workflows, you need to configure Claude Code's GitHub integration.

### Installation Steps

1. **Install the Claude Code GitHub App**

   Run the following command in your Claude Code CLI:
   ```bash
   /install-github-app
   ```

   This command will:
   - Guide you through authorizing the Claude Code GitHub App
   - Set up the required OAuth token
   - Configure the necessary permissions for Claude to interact with your repositories

2. **Configure Repository Secrets**

   After installing the app, you need to add the OAuth token as a repository secret. The `/install-github-app`
   command should do this automatically.
   - Go to your repository settings
   - Navigate to **Secrets and variables** → **Actions**
   - Add a new repository secret named `CLAUDE_CODE_OAUTH_TOKEN`
   - Paste the OAuth token provided during the `/install-github-app` setup

Note, this is billed to your Claude _subscription_.

3. **Add Workflow Files**

   Copy the workflow files to your repository:
   - `.github/workflows/claude-code-review.yml`
   - `.github/workflows/claude.yml`

4. **Customize User Access**

   Update the `ALLOWED_USERS` environment variable in both workflow files to include your GitHub username and any other authorized users.

## Claude Code Workflows Overview

This repository uses two GitHub Actions workflows to integrate Claude Code:

1. **Claude Code Review** (`.github/workflows/claude-code-review.yml`) - Automatic PR reviews
2. **Claude Code** (`.github/workflows/claude.yml`) - On-demand Claude assistance via @mentions

## Claude Code User Access Control

Both Claude Code workflows are restricted to specific users via the `ALLOWED_USERS` environment variable:

```yaml
env:
  ALLOWED_USERS: '["mattjmcnaughton"]'
```

To add additional users, update this array in both workflow files.

## Claude Code Workflow Details

### Workflow 1: Automatic PR Reviews

**File:** `.github/workflows/claude-code-review.yml`

### Triggers
- When a pull request is opened
- When a pull request is synchronized (new commits pushed)

### Behavior
Automatically runs a Claude Code review on every PR from allowed users. The review checks:
- Code quality and best practices
- Potential bugs or issues
- Performance considerations
- Security concerns
- Test coverage

The review is posted as a comment on the PR using `gh pr comment`.

### Configuration
- Uses repository's `CLAUDE.md` for style and convention guidance
- Only has access to specific `gh` commands for safety:
  - `gh issue view`
  - `gh search`
  - `gh issue list`
  - `gh pr comment`
  - `gh pr diff`
  - `gh pr view`
  - `gh pr list`

### Workflow 2: On-Demand Claude Assistance

**File:** `.github/workflows/claude.yml`

#### Triggers
The workflow activates when an allowed user mentions `@claude` in:
- Issue comments
- Pull request review comments
- Pull request reviews
- Issue titles or bodies

#### Behavior
Claude performs the specific instructions provided in the comment/issue that tagged it.

**Examples:**
- `@claude please add unit tests for the authentication module`
- `@claude review this code for security issues`
- `@claude update the documentation to reflect these changes`

#### Configuration
- No predefined prompt - Claude follows the instructions in your message
- Has access to read CI results on PRs (via `actions: read` permission)
- Can be customized with `claude_args` for specific tool restrictions

## Claude Code Required Secrets

Both workflows require the following repository secret:
- `CLAUDE_CODE_OAUTH_TOKEN` - OAuth token for Claude Code authentication

## Claude Code Permissions

Both Claude Code workflows have the following permissions:
- `contents: read` - Read repository contents
- `pull-requests: write` - Read and write pull request data (needed to post comments)
- `issues: write` - Read and write issue data (needed to post comments)
- `id-token: write` - Required for authentication
- `actions: read` - Read CI results (claude.yml only)

## Claude Code Customization

### Restricting to Specific File Paths
You can uncomment the `paths` filter in `claude-code-review.yml` to only trigger on specific file changes:

```yaml
paths:
  - "src/**/*.ts"
  - "src/**/*.tsx"
  - "src/**/*.js"
  - "src/**/*.jsx"
```

### Restricting Available Tools
You can limit what tools Claude can use by setting `claude_args`:

```yaml
claude_args: '--allowed-tools "Bash(gh pr:*)"'
```

## References

- [Claude Code Action Documentation](https://github.com/anthropics/claude-code-action/blob/main/docs/usage.md)
- [Claude Code CLI Reference](https://docs.claude.com/en/docs/claude-code/cli-reference)
