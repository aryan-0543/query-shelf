## Code Quality and Development Guidelines

- Write clean, readable, maintainable Python code. Prefer simplicity over unnecessary abstraction.
- Give each function one clear responsibility. Keep unrelated functionality separate.
- Create separate modules when responsibilities genuinely differ, but avoid unnecessary files, functions, classes, or abstractions.
- Do not add unnecessary dependencies, libraries, variables, configuration files, or code.
- Use clear, descriptive names for functions, variables, modules, and classes.
- Add concise docstrings where useful. Avoid excessive comments; comment only when the reasoning is not obvious.
- Separate configuration, schemas, business logic, and tests when appropriate.
- Follow the existing project structure. Propose structural changes when necessary, but avoid premature complexity.
- Handle errors explicitly and provide useful error messages.
- Write focused tests for important functionality and edge cases.
- Explain non-obvious design decisions and justify new dependencies or abstractions.
- Do not modify unrelated code or introduce features outside the current milestone.
- Implement one approved milestone at a time.
- Run relevant tests and report actual results. Never claim unverified tests passed.
- Keep documentation consistent with the actual implementation and verified progress.
- Before each implementation, read and follow `AGENTS.md`, `ARCHITECTURE.md`, and `IMPLEMENTATION_PLAN.md`.
- After every milestone, provide a concise explanation of changed files, key functions, important design decisions, and any non-obvious code. Do not modify code just to explain it.
- Do not update `README.md`, `ARCHITECTURE.md`, `IMPLEMENTATION_PLAN.md`, or `AGENTS.md` after every code change.
- `AGENTS.md`: Update only when permanent coding standards or development rules change.
- `ARCHITECTURE.md`: Update only when system design, component responsibilities, or data flow changes significantly.
- `IMPLEMENTATION_PLAN.md`: Update when a milestone is completed or the planned sequence changes.
- `README.md`: Update when setup instructions, dependencies, usage, or user-facing functionality changes.
- Fix documentation immediately if a code change makes it inaccurate or misleading.
- At milestone completion, review only the relevant documentation and update only what is necessary.

For every feature implementation, continue to update the existing `scripts/manual_load_pdf.py` script to support manual testing, run relevant automated tests, and report actual results.
