# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Refactored MPI logging setup into a clearer pipeline with dedicated steps for logger initialization, filename generation, log-directory creation, and handler registration.
- Centralized warning logging hookup in the logging base class so `py.warnings` is configured once, removing duplicate warning-handler wiring in downstream classes.
- Made file logging optional by default by using `logdir=None`; log files are now only created when a log directory is explicitly provided.
- Split logger and handler verbosity defaults into separate settings (`LOGGER_LOGLVL="WARNING"` and `HANDLER_LOGLVL="DEBUG"`).
- Updated file-handler naming to use the concrete log filename, improving traceability of per-rank/per-run log outputs.
- Added MPI synchronization around log-directory creation to prevent rank race conditions during startup.

### Fixed
- Corrected duplicate-handler cleanup behavior to remove redundant handlers consistently while preserving the intended active handler instance.
- Updated logging-related tests to match the refactored behavior, including initialization, optional file logging, handler setup flow, and duplicate-handler removal expectations.
