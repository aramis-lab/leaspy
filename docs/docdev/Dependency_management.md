---
orphan: true
---
# Dependency Management & Security Policy

## Overview

Leaspy uses **Poetry** for dependency management and packaging. This ensures reproducible builds, precise control over library versions, and strict separation of development vs. production dependencies.

## Core Philosophy: Conservative Updates

We prioritize **stability** and **backward compatibility** over using the absolute latest versions of dependencies.

*   **Avoid "Update All":** We do not blindly run `poetry update` to upgrade all packages at once. This frequently introduces silent breaking changes.
*   **Targeted Fixes:** We update packages primarily to:
    1.  Fix specific security vulnerabilities (CVEs).
    2.  Support new required Python versions.
    3.  Access specific new features needed by Leaspy.

## Security Vulnerability Workflow

When a security vulnerability is identified (e.g., by Dependabot or manual audit):

1.  **Diagnose Risk:** Determine if Leaspy is actually vulnerable.
    *   *Example:* A vulnerability in a web server component of a library is irrelevant if we only use its math functions.
    *   If the risk is negligible and the update carries high regression risk (e.g., PyTorch), we may accept the risk rather than update.

2.  **Targeted Update:** Update *only* the affected package to the lowest secure version.
    ```bash
    # Update a single package to the latest allowed version
    poetry update <package_name>
    ```

3.  **Traceability (The "Lost Forest" Solution):**
    *   Do **not** create permanent markdown files for temporary security audits.
    *   Instead, create a **GitHub Issue** detailing the CVEs, the decision process, and the versions fixed.
    *   Reference this issue in your PR/Commit message.
    *   **Close the issue immediately.** It serves as a permanent, searchable record without cluttering the active backlog.

4.  **Validation:**
    Run the test suite to confirm the update didn't break functionality.
    ```bash
    pytest
    ```

## Specific Package Policies

### PyTorch (`torch`)
We treat PyTorch updates with extreme caution due to its history of breaking API changes in minor releases.

*   **Policy:** We explicitly pin strict upper bounds (e.g., `>=2.2.0,<2.8`).
*   **Rationale:** Preventing accidental upgrades ensures that models trained on one version remain loadable on another, and that numerical stability is preserved.

## Developer protocol

### Initial Setup
```bash
conda create -n leaspy python=3.9
conda activate leaspy
poetry install
```

### Adding a Dependency
```bash
# Production dependency
poetry add <package_name>

# Dev/Doc dependency
poetry add --group dev <package_name>
```

### Syncing Lock File
If you verify or edit `pyproject.toml` manually, you must update the lock file or CI will fail.
```bash
poetry lock
```

### Verifying Security
Check that the lock file is valid and consistent, it is normal if nothing is shown, only errors or warnings will be shown if they exist:
```bash
make check.lock
```

### Running Tests
Run the full test suite to verify changes before commiting:
```bash
pytest
```
