# MultiTalk Worker Fixes

## Issue 1: Multiple Concurrent Instances ✅

**Problem**: Worker was starting with `concurrency: 32 (prefork)`, creating many parallel tasks instead of processing sequentially.

**Solution**: Launch the worker with `--concurrency 1` to run a single instance:

```bash
# Current (32 concurrent workers):
lingo launch lingo_video_worker:lang

# Fixed (1 worker only):
lingo launch lingo_video_worker:lang worker --concurrency 1
```

**Why this happens**: Celery defaults to using the number of CPU cores as the concurrency level. By adding `--concurrency 1`, you tell Celery to use only 1 worker process.

---

## Issue 2: Reference Serialization Error ✅ + Enhanced Debugging

**Problem**: Tasks were failing with `AttributeError: 'dict' object has no attribute 'dump_file'` when `target` parameter should be a `Reference[bytes]` object but was received as a plain dict.

**Solution**: Multiple improvements have been made:

1. **Enhanced Reference Deserialization** (`lingo/celery/language.py`):
   - Detects when a dict is missing the `__lingo_kind__` serialization marker
   - Attempts to convert it using the proper protocol deserialization
   - Logs detailed debug information when conversion occurs
   - This ensures Reference objects are properly reconstructed even if the serialization marker is missing

2. **Debug Flag Added** (`lingo/cli.py` + `lingo/celery/language.py`):
   - Use `lingo launch lingo_video_worker:lang --debug` to enable debugging
   - Prints detailed task execution metadata including:
     - Job ID
     - Task name
     - Celery task ID
     - Attempt number
     - Reference coercion attempts and results

**Files Modified**:
- `lingo/cli.py` - Added `--debug` flag and environment variable passing
- `lingo/celery/language.py` - Enhanced `_coerce_value` method with Reference handling and debug logging

---

## Using the Fixes

### Test 1: Concurrency (Single Worker)
Verify the worker is using only 1 concurrent process:

```bash
conda activate multitalk_blackwell
cd /mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/video_generators/multitalk
lingo launch lingo_video_worker:lang worker --concurrency 1
```

Look for: `[config] .> concurrency: 1` in the output

### Test 2: Debug Mode (Detailed Logging)
Enable debug logging to see detailed task execution information:

```bash
conda activate multitalk_blackwell
cd /mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/video_generators/multitalk
lingo launch lingo_video_worker:lang --debug worker --concurrency 1
```

This will show logs like:
```
[DEBUG] Task started: job_id=..., task=kokoro_tts, attempt=1, celery_task_id=...
[DEBUG] Attempting to coerce dict to Reference: dict_keys=['path', 'content_type', ...], has_path=True, has_marker=False
[DEBUG] Successfully coerced dict to Reference: path=/tmp/...
```

### Test 3: Reference Handling
Run the built-in test to verify Reference objects are handled correctly:

```bash
conda activate multitalk_blackwell
cd /mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/video_generators/multitalk
lingo test lingo_video_worker.py
```

This will run `test_kokoro_tts_conversation` and `test_multitalk_pipeline_conversation` which test Reference object handling.

---

## Debugging with the New Flag

When you encounter issues, run with the `--debug` flag to see:
- When tasks start and their metadata
- Reference dict-to-object conversion attempts
- Success/failure of serialization recovery
- Exact value types being passed to handlers

Example:
```bash
lingo launch lingo_video_worker:lang --debug worker --concurrency 1 2>&1 | grep DEBUG
```

---

## Summary

- **Issue 1 (Concurrency)**: Add `worker --concurrency 1` to the launch command ✅
- **Issue 2 (Reference)**: Fixed in lingo's `_coerce_value` method with automatic recovery ✅
- **New Feature**: `--debug` flag for enhanced diagnostic logging ✅
- All fixes are backward compatible and don't affect other functionality

