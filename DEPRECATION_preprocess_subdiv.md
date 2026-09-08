# Deprecation Plan: `preprocess_subdiv` Option

## Overview
The `preprocess_subdiv` option is being deprecated due to potential for unexpected behavior and MPI deadlocks when set to `True`. This document outlines the deprecation timeline and removal plan.

## Problem Statement
- When `preprocess_subdiv=True`, preprocessing happens on subdivisions rather than the full stream
- This can cause MPI deadlock issues when ranks have different amounts of data
- The option adds significant code complexity with conditional branches
- Best practice is to always preprocess the full stream before subdivision

## Deprecation Timeline

### Phase 1: Current Version (e.g., v0.5.x) ✅
**Status**: Implemented

**Actions**:
- Warn users when `preprocess_subdiv=True` is detected
- Automatically override to `False` for safety
- Add deprecation warnings with version info
- Document in release notes

**Code Changes**:
- Added `DeprecationWarning` in `__init__` when option is `True`
- Always set to `False` internally
- Added `TODO` comments marking code to remove in v1.0.0

**User Impact**: 
- Users see a warning if they set `preprocess_subdiv=True`
- No breaking changes - code continues to work
- Users should remove the option from their config files

### Phase 2: Next Minor Version (e.g., v0.6.0)
**Status**: Planned

**Actions**:
- Warn even when `preprocess_subdiv=False` is explicitly set
- Update warning message: "Option is deprecated and will be removed in v1.0.0"
- Keep all code paths for backward compatibility
- Update all example configurations to remove the option
- Update documentation extensively

**Code Changes**:
```python
if "preprocess_subdiv" in options["co"]:
    warn(
        "The 'preprocess_subdiv' option is deprecated and will be removed "
        "in version 1.0.0. Please remove it from your configuration. "
        "Preprocessing now always happens on the full stream.",
        DeprecationWarning,
        stacklevel=2
    )
    options["co"]["preprocess_subdiv"] = False
```

**User Impact**:
- All users with the option in config files see a warning
- No breaking changes - code continues to work
- Strong encouragement to update configuration files

### Phase 3: Next Major Version (e.g., v1.0.0)
**Status**: Planned

**Actions**:
- Remove the option completely
- Remove all conditional code branches
- Simplify the preprocessing logic
- Clean up tests

**Code to Remove**:
1. In `__init__`: Remove all handling of `preprocess_subdiv`
2. In `_generate_data` (line ~738): Remove the check `if not self.ex_dict and self.options["preprocess_subdiv"]:`
3. In `_generate_data` (line ~842): Remove `if not self.options["preprocess_subdiv"]:` - always do preprocessing
4. In `_generate_data` (line ~901): Remove the entire `if self.options["preprocess_subdiv"]:` block
5. In `corr_hdf5.py`: Remove from list of valid options

**Simplified Code**:
The preprocessing in `_generate_data` becomes simply:
```python
# Preprocessing always happens on full stream before subdivision
try:
    self.logger.debug("Preprocessing read_len stream...")
    st = preprocess_stream(
        st, self.store_client, startt, endt, tl, **self.options
    )
    self.logger.debug("Finished preprocessing read_len stream.")
except Exception as e:
    self.logger.error(
        f"Stream preprocessing failed for time {t} and stream {st}.\n"
        f"The Original Error Message was {e}."
    )
    st = Stream()
```

**User Impact**:
- **BREAKING CHANGE**: Config files with `preprocess_subdiv` will cause error
- Users must remove the option from their configuration
- Error message should guide users: "Unknown option 'preprocess_subdiv'. This option was removed in v1.0.0. Please update your configuration file."

## Migration Guide for Users

### For v0.5.x → v0.6.0 (Non-breaking)
Update your parameter file or configuration dictionary:

**Before**:
```yaml
co:
  preprocess_subdiv: True  # or False
  # ... other options
```

**After**:
```yaml
co:
  # preprocess_subdiv removed - preprocessing always on full stream
  # ... other options
```

### For v0.6.x → v1.0.0 (Breaking)
If you still have `preprocess_subdiv` in your config:
1. Remove the line from your YAML/config file
2. No functional changes needed - behavior is the same
3. If you see an error about unknown option, simply delete that line

## Testing Considerations

### Current (v0.5.x)
- ✅ Tests should set `preprocess_subdiv=False` or not set it at all
- ✅ Add test to verify warning is raised when `True`

### v0.6.0
- Add test to verify warning is raised even when `False`
- Ensure all tests pass with option removed

### v1.0.0
- Remove all mentions of `preprocess_subdiv` from tests
- Verify error handling for unknown options
- Ensure all existing tests pass

## Documentation Updates

### v0.5.x (Current)
- ✅ Add note to parameter documentation
- ✅ Add entry to CHANGELOG

### v0.6.0
- Update all example configuration files
- Add prominent note in documentation about upcoming removal
- Update migration guide

### v1.0.0
- Remove from parameter documentation completely
- Update CHANGELOG with breaking change notice
- Update migration guide for major version

## Rationale

**Why deprecate?**
- Potential for MPI deadlocks when `True`
- Adds unnecessary code complexity
- Single behavior (preprocessing full stream) is simpler and more reliable
- No significant performance benefit to subdivision preprocessing

**Why phased approach?**
- Gives users time to update configurations
- Follows semantic versioning principles
- Minimizes disruption to existing workflows
- Clear communication at each stage

## References
- Related issue: MPI deadlock in `test_generate_exhand`
- Code locations marked with `TODO: Remove in version 1.0.0`
- Semantic Versioning: https://semver.org/
