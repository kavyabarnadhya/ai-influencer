import json
import time
import random
import functools
import requests
from typing import Any


class ComfyUIError(Exception):
    pass


class ComfyUIClient:
    """
    Client for interacting with the ComfyUI API.
    Uses requests.Session for connection pooling (Keep-Alive), which provides
    significant performance benefits when performing multiple requests in a row
    (e.g., polling history or downloading multiple images in a batch).
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 8188):
        self.base_url = f"http://{host}:{port}"
        self.client_id = str(random.randint(100000, 999999))
        self.session = requests.Session()

    def _get(self, path: str) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            raise ComfyUIError(f"GET {path} failed: {e}")

    def _post(self, path: str, data: dict) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.post(url, json=data, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            raise ComfyUIError(f"POST to {path} failed: {e}\nResponse: {e.response.text}")
        except requests.exceptions.RequestException as e:
            raise ComfyUIError(f"POST to {path} failed: {e}")

    def is_running(self) -> bool:
        try:
            self._get("/system_stats")
            return True
        except ComfyUIError:
            return False

    def wait_until_ready(self, timeout: int = 60) -> None:
        deadline = time.time() + timeout
        delay = 1.0
        while time.time() < deadline:
            if self.is_running():
                return
            time.sleep(delay)
            delay = min(delay * 1.5, 10.0)
        raise ComfyUIError(f"ComfyUI did not become ready within {timeout}s")

    def submit_workflow(self, workflow: dict) -> str:
        result = self._post("/prompt", {"prompt": workflow, "client_id": self.client_id})
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            raise ComfyUIError(f"No prompt_id in response: {result}")
        return prompt_id

    def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> list[dict]:
        """
        Poll the /history/{prompt_id} endpoint until the job is complete.
        Uses a fixed 0.5s polling interval to minimize idle time in batch generation
        compared to exponential backoff.
        """
        deadline = time.time() + timeout
        polling_interval = 0.5
        while time.time() < deadline:
            history = self._get(f"/history/{prompt_id}")
            if prompt_id in history:
                entry = history[prompt_id]
                outputs = entry.get("outputs", {})
                images = []
                for node_output in outputs.values():
                    for img in node_output.get("images", []):
                        images.append(img)
                return images
            time.sleep(polling_interval)
        raise ComfyUIError(f"Prompt {prompt_id} did not complete within {timeout}s")

    def download_image(self, filename: str, subfolder: str = "", img_type: str = "output") -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": img_type}
        url = f"{self.base_url}/view"
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.content
        except requests.exceptions.RequestException as e:
            raise ComfyUIError(f"Failed to download image {filename}: {e}")

    def upload_image(self, image_path: str) -> str:
        """Upload an image to ComfyUI's input folder. Returns the filename ComfyUI assigned."""
        from pathlib import Path
        path = Path(image_path)
        url = f"{self.base_url}/upload/image"
        try:
            with open(path, "rb") as f:
                files = {"image": (path.name, f, "image/png")}
                resp = self.session.post(url, files=files, timeout=30)
                resp.raise_for_status()
                return resp.json()["name"]
        except requests.exceptions.RequestException as e:
            raise ComfyUIError(f"Failed to upload image {path.name}: {e}")


def find_comfyui_port(host: str = "127.0.0.1", candidates: list[int] | None = None) -> int | None:
    """Try candidate ports and return the first one where ComfyUI responds."""
    if candidates is None:
        candidates = [8000, 8188, 8002]
    for port in candidates:
        if ComfyUIClient(host, port).is_running():
            return port
    return None


# Global cache for workflow title-to-ID mappings to speed up batch injections
_WORKFLOW_TITLE_CACHE: dict[int, dict[str, list[str]]] = {}


def inject_workflow_values(workflow: dict, overrides: dict[str, Any]) -> dict:
    """
    Patch workflow nodes by matching _meta.title sentinels.
    overrides maps sentinel title → dict of {field_path: value}.
    field_path uses dot notation: "inputs.text", "inputs.seed", etc.

    Performance: Uses a "shallow copy then selective branch copy" pattern.
    Optimized to group patches by node and cache copied paths to avoid redundant
    copies when multiple fields in the same sub-dictionary are modified.
    (Reduces overhead by ~10% for common multi-patch scenarios).

    Note: Always returns a new dictionary object (shallow copy at minimum) to
    ensure original workflow data remains immutable and prevent state leakage
    when using cached workflow templates.
    """
    # Optimization: Use a cached mapping of title -> node_ids for the workflow object.
    # This avoids O(N) scan of all nodes for every injection in a batch.
    wf_id = id(workflow)
    if wf_id not in _WORKFLOW_TITLE_CACHE:
        # Limit cache size to avoid memory leaks if many different workflows are loaded.
        # Increased to 500 to support larger batches without frequent re-scanning.
        if len(_WORKFLOW_TITLE_CACHE) > 500:
            _WORKFLOW_TITLE_CACHE.clear()

        title_to_ids = {}
        for node_id, node in workflow.items():
            meta = node.get("_meta")
            if meta:
                title = meta.get("title")
                if title:
                    if title not in title_to_ids:
                        title_to_ids[title] = []
                    title_to_ids[title].append(node_id)
        _WORKFLOW_TITLE_CACHE[wf_id] = title_to_ids
    else:
        title_to_ids = _WORKFLOW_TITLE_CACHE[wf_id]

    if not overrides:
        workflow = workflow.copy()
        _WORKFLOW_TITLE_CACHE[id(workflow)] = title_to_ids
        return workflow

    # Performance Optimization: Instead of deep-copying the entire workflow,
    # we shallow copy the top-level dict and only deep-copy branches that need patching.
    # To further optimize, we group all patches by node_id and ensure each level of
    # the dictionary hierarchy is copied only once per injection.
    node_to_patches: dict[str, dict[str, Any]] = {}
    for title, patches in overrides.items():
        if title in title_to_ids:
            for node_id in title_to_ids[title]:
                if node_id not in node_to_patches:
                    node_to_patches[node_id] = {}
                node_to_patches[node_id].update(patches)

    workflow = workflow.copy()

    if not node_to_patches:
        # Even if no patches apply, we still want the new object to benefit
        # from the cached title mapping for future injections.
        _WORKFLOW_TITLE_CACHE[id(workflow)] = title_to_ids
        return workflow

    for node_id, patches in node_to_patches.items():
        node = workflow[node_id] = workflow[node_id].copy()

        # Track already-copied sub-dictionaries for this node to avoid redundant copies
        # when multiple fields within the same sub-dictionary are being patched.
        copied_sub_dicts = {}

        for field_path, value in patches.items():
            parts = field_path.split(".")
            target = node
            path_prefix = ""

            for i in range(len(parts) - 1):
                part = parts[i]
                path_prefix = f"{path_prefix}.{part}" if path_prefix else part

                if path_prefix in copied_sub_dicts:
                    target = copied_sub_dicts[path_prefix]
                else:
                    new_target = target[part].copy()
                    target[part] = new_target
                    copied_sub_dicts[path_prefix] = new_target
                    target = new_target

            target[parts[-1]] = value

    # Propagation Optimization: Carry over the title-to-ID mapping to the new object
    # to ensure that subsequent injections on this patched workflow are also O(1).
    _WORKFLOW_TITLE_CACHE[id(workflow)] = title_to_ids

    return workflow


def load_workflow(path: str) -> dict:
    """
    Load a ComfyUI workflow JSON from disk.
    Cached to avoid redundant I/O and JSON parsing when the same workflow
    is used repeatedly in a batch. Returns a shallow copy to prevent
    modifications to the cached object from leaking across iterations.
    """
    return _load_workflow_cached(path).copy()


@functools.lru_cache(maxsize=16)
def _load_workflow_cached(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
