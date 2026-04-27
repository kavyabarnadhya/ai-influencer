## 2025-01-24 - Selective Deep-Copying for Workflow Injection

**Learning:** In Python, `json.loads(json.dumps(obj))` is often faster than `copy.deepcopy(obj)`, but it still has O(N) complexity where N is the size of the entire object. In ComfyUI workflows, which can be large, deep-copying the entire workflow to change only a few nodes is inefficient.

**Action:** Use a "shallow copy then selective deep copy" pattern. Shallow copy the main container (e.g., `workflow.copy()`), and then only deep-copy the specific sub-objects (nodes) that are actually being modified. This reduces the overhead from O(total_workflow_size) to O(modified_nodes_size + num_nodes).
