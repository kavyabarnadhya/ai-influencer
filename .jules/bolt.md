## 2025-01-24 - Selective Deep-Copying for Workflow Injection

**Learning:** In Python, `json.loads(json.dumps(obj))` is often faster than `copy.deepcopy(obj)`, but it still has O(N) complexity where N is the size of the entire object. In ComfyUI workflows, which can be large, deep-copying the entire workflow to change only a few nodes is inefficient.

**Action:** Use a "shallow copy then selective deep copy" pattern. Shallow copy the main container (e.g., `workflow.copy()`), and then only deep-copy the specific sub-objects (nodes) that are actually being modified. This reduces the overhead from O(total_workflow_size) to O(modified_nodes_size + num_nodes).

## 2025-01-25 - Cached Node Mappings and Recursive Shallow Copying

**Learning:** Iterating over all nodes in a large ComfyUI workflow (200+ nodes) to find matching titles is expensive (O(N)) when done repeatedly in batch generations. Additionally, `json.loads(json.dumps(obj))` for node copying is ~30x slower than a manual recursive shallow copy of only the modified path.

**Action:** Implement a title-to-ID mapping cache (using `id(workflow)` as the key) to make node lookups O(1) after the first pass. Use a recursive shallow copy strategy to traverse and clone only the dictionary branches that are being modified, ensuring correctness without the overhead of full serialization.
