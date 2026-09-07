# Local Agent Instructions

- When scanning for optimizations, avoid any that require database migrations or meddle with database commit behavior.
- When scanning for optimizations, ignore FreqAI- and orderflow-related code.
- When scanning for optimizations, do not change or bypass `FtPrecise` calculations.
- Low- and medium-risk optimizations must not require patching existing tests.
- When scanning for optimizations, preserve all fields and data columns exposed by API endpoints (for example, historic balance), even if Freqtrade's own code does not consume them. External API users may depend on that data. Trace the actual API response before treating a field or column as unused.
