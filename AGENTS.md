# Local Agent Instructions

- When scanning for optimizations, avoid any that require database migrations or meddle with database commit behavior.
- When scanning for optimizations, ignore FreqAI- and orderflow-related code.
- When scanning for optimizations, do not change or bypass `FtPrecise` calculations.
- Low- and medium-risk optimizations must not require patching existing tests.
