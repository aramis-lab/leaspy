---
orphan: true
---
# Contribution Guide

This guide explains how to contribute to Leaspy, including setting up your environment, creating pull requests, and handling common scenarios like upstream changes.

## Key Concepts

Before diving in, let's define some Git terminology used throughout this guide:

| Term | Definition |
|------|------------|
| **Fork** | Your personal copy of the Leaspy repository on GitHub. You have full control over it. |
| **upstream** | The official Leaspy repository (`aramis-lab/leaspy`). You can read from it but not write directly. |
| **origin** | Your fork on GitHub. This is where you push your changes. |
| **fetch** | Downloads new commits from a remote (like `upstream`) without modifying your local files. Think of it as "check for updates". |
| **merge** | Combines changes from one branch into another, creating a merge commit that preserves both histories. |
| **checkout** | Switches your working directory to a different branch, or creates a new branch. |

## Repository Structure

Leaspy uses a **fork-based workflow** with the following remotes:

| Remote     | Repository                          | Purpose                        |
|------------|-------------------------------------|--------------------------------|
| `upstream` | `https://github.com/aramis-lab/leaspy` | Official repository (read-only for contributors) |
| `origin`   | Your fork on GitHub (you have to [set up](#initial-setup-first-time-contributors) first )                | Your personal copy (read-write) |

### Branch Naming Convention

When we will create new branches in our fork, we have to follow some naming conventions, the name of your branch will be different depending on what you are going to do.

| Branch Type | Pattern | Example | Description |
|-------------|---------|---------|-------------|
| Version branches | `v{major}.{minor}` | `v2.1`, `v3.0` | Active development for a minor version |
| Feature branches | `feature/{name}` | `feature/plots` | New functionality |
| Fix branches | `fix/{name}` | `fix/vulnerabilities` | Bug fixes |
| Release branches | `release/v{X.Y.Z}` | `release/v2.0.2` | Preparing a specific release |

> **Example:** if we want to fix the csv generation methods, our branch will be called: `fix/csv_generation`.

## Initial Setup (First-Time Contributors)

### 1. Fork the Repository

1. Go to [https://github.com/aramis-lab/leaspy](https://github.com/aramis-lab/leaspy)
2. Click **Fork** (top-right corner)
3. Select your account as the destination

Now in your personal repo, you should have a new repository that is a copy from Leaspy, it is safer to work here because any changes you will do won't affect the original repository, so even the worst case scenario, if you overwride, delete or force remove the repo, it won't touch the original leaspy.

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/leaspy.git
cd leaspy
```

### 3. Add Upstream Remote

```bash
git remote add upstream https://github.com/aramis-lab/leaspy.git
```

### 4. Verify Remotes

```bash
git remote -v
```

Expected output:
```
origin    https://github.com/YOUR_USERNAME/leaspy.git (fetch)
origin    https://github.com/YOUR_USERNAME/leaspy.git (push)
upstream  https://github.com/aramis-lab/leaspy.git (fetch)
upstream  https://github.com/aramis-lab/leaspy.git (push)
```

### 5. Set Up Development Environment

```bash
conda create -n leaspy python=3.11
conda activate leaspy
poetry install
```

## Creating a Pull Request

### Step 1: Identify the Target Branch

Before creating a branch, determine which version branch your contribution targets. This decision affects when your changes will be released to users:

| Contribution Type | Target Branch | Why? |
|-------------------|---------------|------|
| **Critical bug fixes** | `master` | Security patches or breaking bugs need immediate release. Users shouldn't wait. |
| **Regular bug fixes** | Current version branch (e.g., `v2.1`) | Will be included in the next patch release (e.g., `v2.1.1`). |
| **New features** | Next minor version branch (e.g., `v2.2`) | New functionality should wait for a planned minor release to ensure proper testing and documentation. |
| **Documentation** | Same branch as the code it documents | Keeps docs synchronized with the corresponding code version. |

> **Example:** You're fixing a typo in a docstring. The current stable version is `v2.1`. Target `v2.1` so it goes out with the next patch. But if you're adding a new algorithm, target `v2.2` (or whatever the next minor version is) so it gets proper review time.

### Step 2: Sync with Upstream

Always start from a fresh, up-to-date version of the target branch. This ensures you're working with the latest code and minimizes future conflicts.

```bash
# Download the latest commits from the official repository (without modifying your files yet)
git fetch upstream

# Create a new branch based on the upstream version branch
# This ensures your starting point is current, not your potentially outdated local copy
git checkout -b my-feature upstream/v2.1
```

The `checkout -b` command does two things: creates a new branch called `my-feature` and switches to it. The `upstream/v2.1` part tells Git to base this new branch on the upstream's `v2.1` branch (replace with your target version).

### Step 3: Make Your Changes

When commiting your changes, do not forget that when you do `git add .`, you are submiting all the changes present in your local repo. Sometimes you can edit files changes by mistake, or you just forget that you did, all those files will be changed, that's why you need to check was you added with `git status`, if you made a mistake type `git reset` and add what you really need. Somethimes is safer to add manually the files we want to change, you do so adding the files or the folders with `git add filename1 foldername1`.

```bash
# Make changes, then stage and commit
git add .
# or
git add file1 file2 file3

git status
git commit -m "feat: add new plotting functionality"
```

Try to give information about what you did, commits like `git commit -m "changes"` are just useless. You can write the message in the text editor, even including various lines, and copy paste to your terminal. You could ask to chatGPT (specially if you are using copilote) what message you could put with "Help me to write a commit message, you can check the changes", then review message, fix it (it talks about not important changes often), and then commit it.

(Optional) Follow commit message conventions:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `refactor:` for code restructuring
- `test:` for test additions/changes

### Step 4: Push to Your Fork

For the **first push** of a new branch, you need to set up tracking:

```bash
git push -u origin my-feature
```

The `-u` flag links your local branch to the remote branch on your fork. After this initial push, subsequent pushes only require:

```bash
git push
```

### Step 5: Create the Pull Request

1. Go to your fork on GitHub
2. Click **Compare & pull request** (appears after pushing)
3. **CRITICAL**: Verify the base repository and branch:
   - Base repository: `aramis-lab/leaspy`
   - Base branch: The target version (e.g., `v2.1`)
   - Head repository: Your fork
   - Head branch: Your feature branch

4. Fill in the PR template:
   - Clear title describing the change
   - Description of what and why
   - Reference any related issues if needed (`Fixes #123`)

5. Request reviewers
6. Submit the PR

## Handling Upstream Changes

When the official repository changes while you're working on your branch, you need to incorporate those changes. **We always use merge** (not rebase) because it's safer and easier to understand.

When the official repository changes and you have an active branch, you can face one or more of these problems:

[New release while you are working in your branch](#scenario-new-release-while-youre-working)
[New release and your PR is pointing to the old release](#scenario-your-pr-target-branch-changed)
[New mayor release](#scenario-major-upstream-restructuring)

### Scenario: New Release While You're Working

You're working on a feature branch, and a new version is released upstream (e.g., `v2.0.2` is released while you're working on something for `v2.0.x`). This can cause conflicts if the same files were modified.

**Solution: Merge the upstream changes into your branch**

```bash
# 1. First, create a backup branch (safety net)
git branch backup-my-feature

# 2. Download the latest upstream commits
git fetch upstream

# 3. Merge the upstream changes into your branch
git merge upstream/master
```

If conflicts occur, resolve them (see [Resolving Conflicts](#resolving-conflicts)), then:

```bash
git add .
git commit -m "Merge upstream/master into my-feature"
```

After merging, push to update your PR:

```bash
git push
```

> **Why merge instead of rebase?** Merge preserves your commit history exactly as it happened and creates a clear "merge point" in the history. If something goes wrong, you can easily identify when the merge happened and revert if needed. Rebase rewrites history, which can cause confusion and requires force-pushing—risky if you're not experienced with Git.

### Scenario: Your PR Target Branch Changed

You created a PR targeting one version branch (e.g., `v2.0`), but the team decided it should go into a different version (e.g., `v2.1`) instead.

**Solution: Change the PR base branch on GitHub + merge locally**

1. **On GitHub**: Edit the PR and change the base branch to the new target

2. **Locally**: Merge the new target into your branch

```bash
git fetch upstream
git merge upstream/v2.1  # Replace v2.1 with your new target branch
git push
```

This brings your branch up to date with the new target, and your PR will update automatically.

### Scenario: Major Upstream Restructuring

If upstream undergoes significant changes (e.g., a major version is released, branch structure changes):

**Solution: Create a backup, then merge**

```bash
# 1. Create a safety backup (you can return to this if anything goes wrong)
git branch backup-my-feature-before-sync

# 2. Fetch upstream
git fetch upstream

# 3. Merge the target branch
git merge upstream/v2.1  # Replace with your target version

# 4. Resolve any conflicts, then push
git push
```

The backup branch lets you compare what changed or restore your work if the merge doesn't go as expected.

## Resolving Conflicts

### In VS Code

1. When a conflict occurs, VS Code shows affected files in the Source Control panel
2. Open a conflicted file - you'll see conflict markers:
   ```
   <<<<<<< HEAD
   your changes
   =======
   upstream changes
   >>>>>>> upstream/v2.1
   ```
3. VS Code provides buttons above each conflict:
   - **Accept Current Change**: Keep your version
   - **Accept Incoming Change**: Keep upstream's version
   - **Accept Both Changes**: Include both
   - **Compare Changes**: See side-by-side diff

4. After resolving all conflicts in a file, stage it:
   ```bash
   git add <filename>
   ```

5. Continue the merge:
   ```bash
   git commit -m "Merge upstream changes"
   ```

### Common Conflict Scenarios

| Conflict Type | Typical Resolution |
|---------------|-------------------|
| Both modified same line | Manually combine logic |
| File deleted upstream, modified locally | Decide if changes are still needed |
| Both added same file | Merge contents or choose one |

> **Note:** For `poetry.lock` conflicts, accept either version and then run `poetry lock` to regenerate it properly. See [Dependency Management](Dependency_management.md) for more details.

## Code Review Process

### As a Contributor

1. Respond to all review comments
2. Make requested changes in new commits (don't force-push during review unless asked)
3. Re-request review after addressing feedback
4. Once approved, a maintainer will merge

### As a Reviewer

Check the following before approving:

- [ ] Code follows project style. Run `ruff check .` locally to verify. Ruff is a fast Python linter that checks for code style issues and potential bugs (see [Ruff documentation](https://docs.astral.sh/ruff/)).
- [ ] Tests pass (`pytest`)
- [ ] Documentation updated if needed
- [ ] No unrelated changes included
- [ ] Commit messages are clear
- [ ] PR targets the correct branch

## Troubleshooting

### "Cannot merge: You have unstaged changes"

Git won't let you merge if you have uncommitted changes. You have two options:

**Option 1: Commit your changes first**
```bash
git add .
git commit -m "WIP: my current work"
git merge upstream/v2.1
```

**Option 2: Stash your changes temporarily**

Stashing saves your uncommitted changes to a temporary storage, letting you work with a clean state:
```bash
# Save current changes to stash
git stash

# Now you can merge
git merge upstream/v2.1

# Restore your stashed changes
git stash pop
```

### PR Shows Wrong Commits

If your PR shows commits that aren't yours, your branch has diverged from the target. Merge the target into your branch:
```bash
git fetch upstream
git merge upstream/v2.1  # Replace with your target branch
git push
```
