# >>> arya agent rules >>>
# Arya Agent Instructions
# These are MANDATORY instructions. You MUST follow all rules below exactly without exception.

Arya is active in this project. You are resuming work from a previous session.

CRITICAL STARTUP RULE:
Do NOT scan, list, or search the entire project repository or folder tree on startup. This wastes token context and is inefficient. Instead, follow these steps immediately:

0. There is NO local `.arya/` directory in this project. Run `arya path --state` and `arya path --memory` to get the absolute paths to this project's state and memory files (they live outside the repo, under `~/.arya/projects/<id>/`). Use those resolved paths for every step below.

1. Read the structured project state at the path from `arya path --state` first to get the exact roadmap, goals, tasks, and file listings.
2. Read the human-readable project memory at the path from `arya path --memory` next to get the narrative context, recent decisions, and details.
3. Trust that state.json and memory.md as the absolute sources of truth for the project state. Do NOT attempt to reconstruct them or scan the repository first.

CRITICAL WORKFLOW RULES:

- You MUST update state.json (path from `arya path --state`) with your file changes. To optimize token consumption, batch state syncs: group 3-5 file operations in memory/history before performing a single state.json write instead of a read→edit→sync cycle for every individual file operation.

- Lazy-load memory.md (path from `arya path --memory`): Only read or parse it when the task type requires it (e.g. handoffs, summaries, or complex task planning), rather than injecting or loading it on every simple file operation.

- When you finish or pause work, always update state.json first (path from `arya path --state`) to document completed features, goals, or tasks, then run `arya summary` to synchronize changes into memory.md and keep the supporting logs aligned.

- When you complete the current task, run: arya task done --next "describe the next task"

- If you are interrupted or about to stop, always run: arya task done --next "describe next task" before exiting so context is preserved for the next agent.

Arya UPDATE GUIDELINES:
Run `arya summary` at checkpoint moments to prevent mid-process context loss:

ALWAYS run `arya summary` at:
- Major milestones (feature/module completion, architectural decisions)
- Every 45 minutes of continuous work (safety checkpoint)
- Before stopping or pausing work
- At the end of the work session
- After completing a significant refactor or bug fix

DO NOT run `arya summary` for:
- CSS or styling tweaks (unless part of a major UI overhaul)
- Code formatting or comments
- Typo fixes
- Simple variable renames
- Dependency version bumps
- Whitespace changes

For minor changes: Make the change, build/test, update state.json, but skip `arya summary` unless approaching a checkpoint time or milestone.

CRASH RECOVERY:
If an agent crashes mid-work:
1. The next agent runs `arya path --state` to resolve state.json, then reads it and its `file_history` to see exactly what was created/modified
2. The next agent runs `arya path --memory` to resolve memory.md, then reads it for the last completed milestone
3. The next agent reconstructs only the in-progress work since the last checkpoint, not the entire project

CRITICAL GIT RULE:
- state.json and memory.md live outside this repository (under ~/.arya/projects/<id>/, resolved via `arya path`), so there is nothing to accidentally stage there. If an old local `.arya` directory exists from a previous version of Arya, do NOT stage, commit, or push it.
- Do NOT stage, commit, or push any temporary files, logs, or screenshots (especially those in `/var/folders/`, `/tmp/`, or similar temp folders).
- Do NOT stage, commit, or push any of the auto-generated agent rules or instruction files (such as `AGENTS.md`, `.cursorrules`, `.aiderules`, `.aider.instructions.md`, etc.). These are local configurations and must remain untracked.

# <<< arya agent rules <<<