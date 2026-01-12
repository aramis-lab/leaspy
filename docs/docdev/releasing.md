---
orphan: true
---
# Releasing a New Version

This guide outlines the step-by-step process for releasing a new version of Leaspy.

## 1. Determine Release Type

Leaspy follows [Semantic Versioning](https://semver.org/). Strict adherence is mandatory.

*   **Patch (x.y.Z)**: Backward-compatible bug fixes *only*, formatting changes, or docs. (e.g., `1.0.8` -> `1.0.9`)
*   **Minor (x.Y.z)**: New functionality that is backward-compatible. Deprecations are allowed but must not remove functionality. (e.g., `1.0.8` -> `1.1.0`)
*   **Major (X.y.z)**: Incompatible API changes. Breaking changes. (e.g., `1.0.8` -> `2.0.0`)

## 2. Prerequisites

Before starting the release process, ensure your local environment is consistent and all tests pass.

```bash
# 1. Ensure lock file is consistent
make check.lock

# 2. Run the full test suite
pytest
```

> **Warning**
> Do not proceed if there are any failures. A release must be stable.

## 3. Update Version Numbers

You must update the version number in two places:

1.  **`pyproject.toml`**: Update the `version` field in the `[project]` section.
2.  **`src/leaspy/__init__.py`**: Update the `__version__` variable.

## 4. Update Changelog

Update `CHANGELOG.md` to reflect the new release.

1.  Locate the **Unreleased** section (if it exists) or create a new header for the version.
2.  Add a header with the version number and today's date (format: `[X.Y.Z] - YYYY-MM-DD`).
3.  Ensure all notable changes are listed under this version.
4.  Move any "Unreleased" content into this new section or ensure the "Unreleased" section is empty for the next cycle.

**Example Format:**
```markdown
### [2.0.1] - 2025-12-17

- [LICENSE] Fix license metadata and text...
```

## 5. Verify CI Configuration

Ensure the Continuous Integration (CI) workflows (in `.github/workflows/`) are configured to run validation on your new version branches.

*   Check `.github/workflows/test.yaml`.
*   If you are creating a new Major or Minor branch (e.g., `v3`, `v2.2`), update the `branches` list to include it (e.g., add `"v3*"`).

## 6. Branch, Commit, and Review

Create a release branch and submit a Pull Request. **Code review is mandatory** for releases to ensure no accidental files or regressions are included.

```bash
# 1. Create a branch for the release
git checkout -b release/vX.Y.Z

# 2. Stage the modified files
git add pyproject.toml src/leaspy/__init__.py CHANGELOG.md

# 3. Commit with a standard message
git commit -m "Release vX.Y.Z"

# 4. Push the branch and open a PR
git push -u origin release/vX.Y.Z
```

**CRITICAL**: Assign a reviewer. Do not merge until you have at least one approval. The reviewer should check:
*   [ ] Version numbers match in all files.
*   [ ] Changelog is complete and date is correct.
*   [ ] CI tests passed.

## 7. Create Release on GitHub

Once the PR is merged into `master`/`main`:

1.  Go to the [GitHub Releases page](https://github.com/MendezSebastianP/leaspy_2a/releases).
2.  Click **Draft a new release**.
3.  **Choose a tag**: Create a new tag `vX.Y.Z` (ensure it matches the code version). Target: `master`.
4.  **Release title**: `vX.Y.Z`.
5.  **Describe this release**:
    *   Copy the relevant section from `CHANGELOG.md`.
    *   Use the "Generate release notes" button to auto-add the list of PRs (New Contributors, etc.).
    *   Highlight any special deployment notes if necessary.
6.  **Save as Draft**: Click **Save draft**.
7.  **Review**: Share the draft link with the team. Ensure the tag, title, and release notes are correct.
8.  **Publish**: Once reviewed, click **Publish release**.

> **Note**: Publishing the release on GitHub will automatically trigger the workflow to publish the package to PyPI.

## Mandatory Release Protocol

This checklist must be fully completed before clicking "Publish" on GitHub.

- [ ] **Environment**: `make check.lock` and `pytest` passed locally.
- [ ] **Version**: `pyproject.toml` and `src/leaspy/__init__.py` match `X.Y.Z`.
- [ ] **Changelog**: `CHANGELOG.md` updated with date and version.
- [ ] **CI Config**: `test.yaml` triggers cover the new version branch.
- [ ] **Review**: Release PR reviewed and approved by another maintainer.
- [ ] **Merge**: Release PR merged into `master`.
- [ ] **Release Github**: Create a Release in GitHub. Release tag matches `vX.Y.Z`.
