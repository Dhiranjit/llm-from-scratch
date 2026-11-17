# AGENTS.md

## Core Constraints

- **Static Analysis Only (No Execution):** Never run shell commands, execute scripts, spin up environments, or run test suites. Reason through code statically—assume I will handle runtime verification myself.
- **Targeted Context Only (No Broad Scanning):** Start from the files and context I provide. You may list directories and inspect directly imported or referenced files when needed for types, interfaces, or dependencies, but do not read unrelated files across the repo or inspect data files unless I explicitly point you to them.
- **Minimal Scope:** Modify only what is directly required for the immediate request. Avoid unsolicited refactoring, reformatting of untouched code, or speculative abstractions (YAGNI).

## Workflow & Communication

- **Simple/Local Tasks:** Execute immediately without preamble or ceremony.
- **Architectural / Multi-File Changes:** Before writing code, briefly outline the plan in plain language and wait for my approval. Focus on **system flow, state movement, and behavioral impact**—do not list file-by-file or line-by-line diffs.
- **Tone:** Keep explanations concise, direct, and grounded in substance. Do not gloss over critical logic, but skip filler.
