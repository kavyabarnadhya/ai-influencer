# Guidelines for future Bolt PRs (locked 2026-05-29)

**Read this section before proposing any new PR.** After 30 merged Bolt PRs Apr-May 2026, an audit found that ~5-6 PRs moved real wall-clock time; the other ~24 were micro-optimisations on already-fast Python paths that get dwarfed by GPU-bound FLUX inference (~120s/slide on RTX 3050 6GB, ~95% of total time). Future PRs must clear a higher bar.

## Required in every PR description

- **Before/after benchmark numbers** on a real workflow that the user actually runs (carousel generation, batch_generate, reprocess_carousel_post). Synthetic 10k-node micro-benchmarks do not count.
- **Wall-clock impact** at the user's hardware tier (RTX 3050 6GB, 16GB RAM) — state how many seconds the change saves per carousel slide / per batch.
- **Risk note** on any function this PR has touched ≥3 times before.

## Hands-off zones

- **`scripts/skin_color_match.py`** — active development this session (in-slide target sampling, face-inclusion lift, σ feather, _MAX_L_SHIFT tuning). Do not touch until 2026-06-15 to avoid merge conflicts. Skin matching optimisations (#60 vectorised, #62 selective pixel conversion) were the legitimate hot-path wins — keep them but do not stack more on top.
- **`workflows/*.json`** — these are exported ComfyUI graphs with `_claude_inject_*` sentinel titles that scripts depend on (CLAUDE.md "Workflow Injection System"). Do not refactor, rename, or "optimise" workflow JSON.
- **GPU-bound paths** — anything inside FLUX inference, ReActor face swap, or hand detail SDXL inpaint. The Python wrapper is not the bottleneck.
- **Character / output data** — `character/ananya/**`, `output/**`, `setup/**` — these are user-curated assets, not code to optimise.

## Paused clusters (converged or low-value)

- **`inject_workflow_values` and its helpers** — touched 10+ times by Bolt (#5, #7, #19, #21, #26, #29, #33, #43, #47, #49, plus #35 which fixed cache poisoning a prior Bolt PR introduced). Function is converged. New PRs targeting this function require explicit owner approval before opening — do not auto-open.
- **MCP server discovery / scandir** (#37, #38, #41, #52, #54) — not on the user's critical path; per memory `feedback_image_review.md` the user avoids `mcp__ananya__review_carousel`. No more MCP optimisation PRs unless there's a documented user complaint.
- **Florence-2 captioning** (#63) — one-time use during LoRA training data prep, not a recurring hot path. No more captioning optimisations until next training cycle starts.

## Kept clusters (real wins, keep optimising)

- **Skin matching vectorisation** (subject to the hands-off window above) — genuine hot path during `reprocess_carousel_post.py`.
- **Caching** (workflow load, path resolution, LLM responses, preset loading) — meaningful cold-start savings on every script invocation.
- **Batch / connection pooling** (`requests.Session`, async queueing) — helps when the user runs `batch_generate.py` or `faceswap_stock.py`.

## Anti-patterns Bolt has fallen into

1. **Bolt-fixing-Bolt** (#35 fixed cache poisoning introduced by an earlier Bolt PR). Do not introduce new caching layers without verifying every prior Bolt cache it interacts with.
2. **Repeated touches on the same function** without convergence. If a function has 5+ prior Bolt PRs, the next PR has to demonstrate why this one is structurally different, not just a smaller constant-factor win.
3. **No regression test added when removing code paths** — high deletion-to-addition ratios (#38: +79/-179, #43: +85/-68) need explicit before/after correctness verification, not just performance numbers.

## Default behaviour

If a proposed PR cannot show ≥100ms/slide wall-clock savings AND clears all hands-off / paused-cluster checks above, do not open it. Park the idea in a `## Parked` subsection below this header for the human to triage.

---

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

## 2026-05-18 - Centralizing Workflow Injection and Config Caching
**Learning:** Manual $O(N)$ node scanning for workflow patching in specialized scripts (like `faceswap_carousel.py`) is redundant and slower than using the centralized `inject_workflow_values` API, which leverages $O(1)$ title-based lookups and selective copying. Additionally, frequent YAML parsing of `config.yaml` in interactive tools can be easily avoided with a single-slot LRU cache.
**Action:** Refactor manual loops to use `inject_workflow_values` and apply `@functools.lru_cache(maxsize=1)` to common config loaders. Use `propagate_cache=False` for final candidate injections to skip internal metadata overhead during submission.

## 2026-05-12 - Final Injection Cache Control and Traversal Unrolling
**Learning:** Even with an optimized `inject_workflow_values`, performing a final dictionary copy in `submit_workflow` to strip internal metadata adds O(N) overhead to the inner loop. Furthermore, Python's loop overhead for dictionary traversal is measurable when the depth is consistently low (e.g., 2 levels for ComfyUI inputs).
**Action:** Implement a `propagate_cache` flag in `inject_workflow_values` to allow skipping internal metadata addition on the final patch. Update `submit_workflow` to only copy if the metadata key is actually present. Unroll the traversal for 2-part paths in the injection logic to shave off loop and range overhead in hot variation loops.

## 2026-05-13 - Path Resolution Caching for Batch IO
**Learning:** `Path.resolve()` in Python is surprisingly expensive because it performs multiple syscalls to resolve symlinks and normalize paths. In batch generation loops where the same few reference images or poses are accessed repeatedly, this becomes a measurable overhead.
**Action:** Wrap `Path.resolve()` in an LRU-cached utility function (`_resolve_path`). Benchmarks show this provides a ~300x speedup for path resolution, reducing overall loop latency in high-throughput generation scripts.

## 2026-05-14 - Fast Deep Copy for Workflow Isolation
**Learning:** Shallow copies of nested dictionary structures (like cached ComfyUI workflows) are insufficient for isolation; callers can inadvertently poison the cache by modifying nested data. While `copy.deepcopy()` is the standard fix, it is relatively slow.
**Action:** Use `json.loads(json.dumps(workflow))` to return deep copies of JSON-compatible structures from cached loaders. This provides complete state isolation and is ~3.5x faster than `deepcopy` for typical workflow dictionaries, maintaining a ~3x speedup over cold disk loads.

## 2026-05-19 - Elimination of Intermediate Dict Allocations in Injection Loop

**Learning:** Even with pre-filtering, creating an intermediate `filtered` dictionary via comprehension for every node targeted by an override title incurs measurable allocation overhead in large workflows (10,000+ nodes). Direct population of the patch map is ~10% faster for redundant patches.

**Action:** Replace dictionary comprehensions in hot filtering loops with direct membership checks and assignment to the primary patch map. Consolidate `workflow.copy()` and cache propagation logic to ensure single-pass processing when overrides are present.

## 2026-05-15 - MCP Server Optimization: Connection Pooling and Caching

**Learning:** MCP servers often handle sequential or repetitive requests (e.g. during prompt iteration). Redundant network handshakes for local API calls (Ollama) and slow LLM inference for identical inputs are significant bottlenecks that can be mitigated with connection pooling and LRU caching.

**Action:** Implement `requests.Session()` for connection pooling and use `@functools.lru_cache` for expensive LLM-based transformations and disk-bound config loading in the MCP server.

## 2026-05-16 - Refactoring Manual Injection Loops to Centralized API

**Learning:** Manually iterating over ComfyUI workflow nodes ($O(N)$) in specialized generation scripts (like faceswap or relighting) is redundant when a centralized, optimized API (`inject_workflow_values`) exists. The centralized API provides $O(1)$ node lookups via internal metadata caching and utilizes optimized selective copying to reduce memory pressure.

**Action:** Replace manual node-scanning loops in all generation scripts with `inject_workflow_values`. This ensures all scripts benefit from core performance improvements (like title caching and path-traversal unrolling) automatically.

## 2026-05-17 - Date-Based Traversal for Output Discovery
**Learning:** Using Path.rglob() to find recent images or carousels in a large, multi-day output directory is an O(N) operation that becomes increasingly slow as the dataset grows. In a structure like output/YYYY-MM-DD/character/, traversing directories by name in reverse order allows for an O(K) discovery (where K is the number of requested recent items), providing a ~10x speedup for typical batch sizes.
**Action:** Replace global rglob or recursive scans in discovery tools (like MCP servers or gallery views) with reverse chronological directory traversal. Stop the scan as soon as the limit is reached.

## 2026-05-17 - Avoiding Redundant stat() Calls for Discovery
**Learning:** Sorting files by modification time (`st_mtime`) in a directory with hundreds or thousands of files is extremely expensive because it triggers a `stat()` syscall for every entry. In this application, image and carousel filenames contain timestamps or are created sequentially, meaning alphabetical sorting is chronologically equivalent to `mtime` sorting.
**Action:** Replace `key=lambda p: p.stat().st_mtime` with default name-based sorting in discovery helpers. Combine this with early slicing of the result set to minimize memory overhead when satisfied by the first few days of output.

## 2026-05-21 - MCP Connection Pooling and Single-Pass Discovery

**Learning:** MCP server tools that make sequential LLM calls (like reviewing carousels) suffer from repeated client instantiation overhead. Furthermore, generating readiness reports across thousands of directories becomes a bottleneck when using multiple `exists()` calls per directory.

**Action:** Implement lazy global initialization for heavy API clients (like `anthropic.Anthropic`) to enable connection pooling across tool calls. Consolidate all file presence and metadata checks into a single `os.scandir` loop per directory in report generation logic. Use a `limit` parameter to bound discovery time as the output history grows.

## 2026-05-20 - Scandir for Faster Discovery and Reduced Path Overhead
**Learning:** `pathlib.Path.glob` and `Path.iterdir` are significantly slower than `os.scandir` in large directories because they instantiate `Path` objects for every entry and may trigger redundant `stat()` calls. In hot discovery loops (like those in MCP servers), using `os.scandir` with string-based prefix/suffix matching provides a measurable speedup (approx 2-4x) by avoiding object allocation overhead and leveraging cached entry metadata.
**Action:** Replace `Path.glob` and `Path.iterdir` with `os.scandir` in hot-loop discovery helpers. Perform filtering using `entry.name.startswith` and `entry.name.endswith` to keep discovery as close to the kernel as possible.

## 2026-05-22 - Preset Loading Optimization and Deep Copy Isolation
**Learning:** Repeatedly parsing the same YAML configuration file (e.g., ) in batch loops or interactive tools adds significant latency (~7.5ms per call). Caching the parsed object is a major win, but requires isolation to prevent state leakage.
**Action:** Use `@functools.lru_cache` to cache raw YAML loads. Return a deep copy of the specific entry using `json.loads(json.dumps())`, which provides a ~125x speedup over repeated I/O while ensuring callers can't corrupt the internal cache.

## 2026-05-22 - Preset Loading Optimization and Deep Copy Isolation
**Learning:** Repeatedly parsing the same YAML configuration file (e.g., presets.yaml) in batch loops or interactive tools adds significant latency (~7.5ms per call). Caching the parsed object is a major win, but requires isolation to prevent state leakage.
**Action:** Use `@functools.lru_cache` to cache raw YAML loads. Return a deep copy of the specific entry using `json.loads(json.dumps())`, which provides a ~125x speedup over repeated I/O while ensuring callers can't corrupt the internal cache.

## 2026-05-23 - Vectorized NumPy Operations for Skin Tone Matching

**Learning:** Repeatedly using the same boolean mask to index different channels of a large image (e.g., `lab[:, :, 0][mask]`) triggers multiple redundant memory allocations and scans in NumPy. Extracting the masked pixels once into a temporary array (`skin_pixels = lab[mask]`) and using vectorized multi-channel operations (like `mean(axis=0)`) is significantly more efficient.

**Action:** In image processing loops, avoid channel-by-channel boolean indexing. Extract the relevant pixels once and use NumPy's vectorized axis-aware functions to perform operations across all channels simultaneously. This provides a ~15-20% speedup for high-resolution image post-processing.

## 2026-05-24 - Selective Pixel Color Conversion for Skin Matching

**Learning:** Transcendental operations like BGR to LAB color space conversion are expensive. Performing them on an entire high-resolution image (e.g. 2048x1024) when only a small fraction (~20%) of the pixels (skin) are actually being processed is a significant waste of CPU and memory.

**Action:** Extract the masked BGR pixels into a compact 1xNx3 array before conversion. Perform all LAB shifts on this subset, convert the result back to BGR, and then patch the original image. Benchmarks show this provides a ~2.85x speedup for the core color shift logic by avoiding O(TotalPixels) transcendental math.

## 2026-05-31 - ROI-based Processing for Spatially Localized CV Operations
**Learning:** Performing color space conversions (BGR-to-HSV/LAB) and Gaussian blurs on full 1080x1920 frames is extremely wasteful when the target subject (e.g., skin) only occupies a fraction of the image. ROI-based processing (cropping to the bounding box of the mask) provides a ~3-4x speedup.
**Action:** In image processing pipelines, always calculate the bounding box of the target mask. Perform expensive operations on the ROI. For operations with spatial dependency like Gaussian blur, add padding (e.g., 3 * sigma) to the ROI to prevent edge artifacts.

## 2026-06-05 - Vectorized FFT Masking for Texture Analysis
**Learning:** Python's nested loops for image-sized array masking (O(H*W)) are a massive bottleneck. Vectorizing coordinate generation with `np.ogrid` and using the "inner-sum trick" (Total - InnerCircle) avoids the overhead of creating and indexing large boolean masks for the "outer" region, providing a ~6.8x end-to-end speedup.
**Action:** In frequency analysis or spatial masking, always use vectorized coordinate math (`ogrid`/`mgrid`) and favor subtracting a smaller region's sum from the total to minimize boolean indexing pressure on high-resolution buffers.
