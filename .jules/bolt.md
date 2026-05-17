## 2025-01-24 - Selective Deep-Copying for Workflow Injection

**Learning:** In Python, `json.loads(json.dumps(obj))` is often faster than `copy.deepcopy(obj)`, but it still has O(N) complexity where N is the size of the entire object. In ComfyUI workflows, which can be large, deep-copying the entire workflow to change only a few nodes is inefficient.

**Action:** Use a "shallow copy then selective deep copy" pattern. Shallow copy the main container (e.g., `workflow.copy()`), and then only deep-copy the specific sub-objects (nodes) that are actually being modified. This reduces the overhead from O(total_workflow_size) to O(modified_nodes_size + num_nodes).

## 2025-01-25 - Cached Node Mappings and Recursive Shallow Copying

**Learning:** Iterating over all nodes in a large ComfyUI workflow (200+ nodes) to find matching titles is expensive (O(N)) when done repeatedly in batch generations. Additionally, `json.loads(json.dumps(obj))` for node copying is ~30x slower than a manual recursive shallow copy of only the modified path.

**Action:** Implement a title-to-ID mapping cache (using `id(workflow)` as the key) to make node lookups O(1) after the first pass. Use a recursive shallow copy strategy to traverse and clone only the dictionary branches that are being modified, ensuring correctness without the overhead of full serialization.

## 2025-01-26 - Grouping Patches and Caching Path Traversal

**Learning:** When multiple fields within the same node or sub-dictionary are patched in one `inject_workflow_values` call, the naive traversal logic performs redundant `copy()` calls and dictionary lookups for the shared parent paths.

**Action:** Group overrides by `node_id` first. During traversal of each node, use a local cache (`copied_sub_dicts`) to track already-copied dictionary branches. This ensures that each level of the dictionary hierarchy is shallow-copied at most once per function call, reducing redundant object creation and memory overhead by ~10% in common multi-patch scenarios (like updating seed, width, height, and prompt in one go).

## 2025-01-27 - Connection Pooling and Loop Optimization
**Learning:** For batch image generation, the overhead of TCP handshakes (using urllib) and exponential polling backoff (up to 15s) can waste minutes of time per batch. Additionally, re-calculating static workflow patches inside the inner loop is redundant.
**Action:** Use `requests.Session` for connection pooling and Keep-Alive. Implement a low, fixed polling interval (e.g., 1.5s) for job completion. Pre-patch constant workflow values outside of generation loops to minimize dictionary operations.

## 2025-01-28 - Caching Workflow Templates with Mandatory Copying
**Learning:** Redundant disk I/O and JSON parsing for the same workflow files (e.g., `t2i_sdxl_upscale.json`) during large batches is a significant bottleneck, but using `@functools.lru_cache` directly on functions returning mutable dicts leads to "state leakage" (cache poisoning) if the objects are later modified.
**Action:** Implement a private cached loader (`_load_workflow_cached`) and a public wrapper (`load_workflow`) that returns a `.copy()`. Also ensure `inject_workflow_values` always returns a new object. This pattern achieves >1000x faster loading for warm caches while maintaining complete state isolation between iterations.

## 2025-01-29 - Latency Hiding through Queuing and Cache Propagation
**Learning:** Polling and downloading images sequentially in batch generations causes significant GPU idle time. Furthermore, caching title-to-ID mappings by object ID is broken when every function call returns a new copy.
**Action:** Implement an asynchronous queuing strategy in all generation scripts—submitting all variations before polling—to keep the GPU saturated. In `inject_workflow_values`, propagate the title-to-ID mapping to the newly created (patched) objects, ensuring that successive injections on a variation remain O(1) instead of re-scanning the entire workflow. Reduce polling intervals to 0.5s for faster response times.

## 2025-01-30 - LRU Cache for Workflows and Identity-Based Path Traversal

**Learning:** Using a "clear all" strategy for global caches in high-throughput batch processes leads to periodic "performance cliffs" where all warm mappings are lost. Furthermore, building string keys for tracking modified dictionary branches in a recursive traversal is ~40% slower than using object identity (`id()`) of the copied sub-objects.

**Action:** Replace simple dict caches with `collections.OrderedDict` implementing an LRU strategy to preserve long-lived "base" workflow mappings while evicting transient variations. In nested object traversal, use a local `copied_sub_dicts` map keyed by the `id()` of newly created copies to avoid redundant string operations and ensures each branch is cloned at most once per call.

## 2025-01-31 - Propagating Title Cache in Workflow Loader

**Learning:** Even with an LRU cache for workflow title-to-ID mappings, `load_workflow` returning a new `.copy()` was causing a "cache cold start" (O(N) scan) for the very first injection on every newly loaded workflow.

**Action:** Extract title scanning logic and update `load_workflow` to pre-scan the base cached workflow and explicitly propagate the mapping to the returned copy's `id()`. This ensures that every workflow returned by the loader starts with a warm cache, making the first injection O(1).

## 2025-02-01 - Redundant Workflow Patching and Connection Pooling
**Learning:** Re-patching constant values (upscalers, LoRAs, prompts) in the inner loop of batch/carousel scripts adds ~35% overhead to `inject_workflow_values` due to redundant dictionary copying and traversal. Additionally, repeated local API calls (Ollama) suffer from unnecessary TCP handshake latency.
**Action:** Move all constant patches out of variation loops and only inject changing values (like seeds) in the inner loop. Use `requests.Session` globally in `prompt_assistant.py` to enable connection pooling for sequential Ollama requests.

## 2026-05-08 - Safe Workflow Title Caching and Polling Optimization
**Learning:** Using `id(dict)` as a global cache key is fundamentally unsafe in Python because memory addresses are reused after garbage collection, leading to "identity collision" bugs. Furthermore, for local ComfyUI instances, a 0.5s polling interval can waste up to 80% of total turnaround time for fast nodes.
**Action:** Avoid global caches keyed by `id()` for mutable objects like workflows. Instead, focus on reducing polling latency (e.g., to 0.2s) and implementing O(1) early returns for no-op injections.

## 2026-05-09 - Workflow Title Caching via Internal Metadata
**Learning:** Repeatedly scanning large ComfyUI workflows for node titles during injection is a significant O(N) bottleneck in batch loops. Storing the mapping in the workflow dictionary itself (and stripping it before API submission) provides O(1) lookups without breaking the API or requiring global state.
**Action:** Implement a "hidden" cache key (e.g., `_claude_title_cache`) in mutable data structures that are passed through transformation pipelines. Ensure the cache is propagated during copies and cleaned up before the data reaches external sinks.

## 2026-05-10 - Redundant Patch Filtering for Workflow Injections
**Learning:** Performing dictionary copies and recursive traversal for patches that do not actually change the underlying workflow data accounts for ~20-50% of the execution time in `inject_workflow_values`. This is common when base overrides are applied multiple times or when "default" values are explicitly patched.
**Action:** Implement a pre-filtering step in `inject_workflow_values` that uses a path-aware redundancy check (`_is_patch_redundant`). This ensures that only patches that actually modify the state trigger a node copy, providing O(1) performance for redundant overrides.

## 2026-05-11 - LLM Response Caching and Immutable Cache Objects
**Learning:** Sequential calls to local LLMs (Ollama) are a massive bottleneck in generation pipelines, often taking seconds per call. Caching these responses is high-impact. Additionally, cached functions returning mutable objects (like lists) are a safety risk; returning immutable tuples is faster and prevents cache corruption.
**Action:** Apply `@functools.lru_cache` to Ollama-dependent prompt polishing functions. Ensure low-level utility functions like `_split_path` return immutable tuples instead of lists to safeguard the cache and slightly improve memory efficiency.

## 2026-05-12 - Final Injection Cache Control and Traversal Unrolling
**Learning:** Even with an optimized `inject_workflow_values`, performing a final dictionary copy in `submit_workflow` to strip internal metadata adds O(N) overhead to the inner loop. Furthermore, Python's loop overhead for dictionary traversal is measurable when the depth is consistently low (e.g., 2 levels for ComfyUI inputs).
**Action:** Implement a `propagate_cache` flag in `inject_workflow_values` to allow skipping internal metadata addition on the final patch. Update `submit_workflow` to only copy if the metadata key is actually present. Unroll the traversal for 2-part paths in the injection logic to shave off loop and range overhead in hot variation loops.

## 2026-05-13 - Path Resolution Caching for Batch IO
**Learning:** `Path.resolve()` in Python is surprisingly expensive because it performs multiple syscalls to resolve symlinks and normalize paths. In batch generation loops where the same few reference images or poses are accessed repeatedly, this becomes a measurable overhead.
**Action:** Wrap `Path.resolve()` in an LRU-cached utility function (`_resolve_path`). Benchmarks show this provides a ~300x speedup for path resolution, reducing overall loop latency in high-throughput generation scripts.

## 2026-05-14 - Fast Deep Copy for Workflow Isolation
**Learning:** Shallow copies of nested dictionary structures (like cached ComfyUI workflows) are insufficient for isolation; callers can inadvertently poison the cache by modifying nested data. While `copy.deepcopy()` is the standard fix, it is relatively slow.
**Action:** Use `json.loads(json.dumps(workflow))` to return deep copies of JSON-compatible structures from cached loaders. This provides complete state isolation and is ~3.5x faster than `deepcopy` for typical workflow dictionaries, maintaining a ~3x speedup over cold disk loads.

## 2026-05-15 - MCP Server Optimization: Connection Pooling and Caching

**Learning:** MCP servers often handle sequential or repetitive requests (e.g. during prompt iteration). Redundant network handshakes for local API calls (Ollama) and slow LLM inference for identical inputs are significant bottlenecks that can be mitigated with connection pooling and LRU caching.

**Action:** Implement `requests.Session()` for connection pooling and use `@functools.lru_cache` for expensive LLM-based transformations and disk-bound config loading in the MCP server.
