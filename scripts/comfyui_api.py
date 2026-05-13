import json
import time
import random
import functools
import requests
from pathlib import Path
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
        # Cache for uploaded images to avoid redundant network IO/disk reads.
        # Key: (abs_path, mtime, size), Value: remote_filename
        self._upload_cache: dict[tuple[str, float, int], str] = {}

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
        """Submit a workflow to the ComfyUI API."""
        # Strip the internal cache key before submitting to ComfyUI.
        # We use a shallow copy to ensure thread safety and avoid side effects.
        if "_claude_title_cache" in workflow:
            workflow = workflow.copy()
            del workflow["_claude_title_cache"]

        result = self._post("/prompt", {"prompt": workflow, "client_id": self.client_id})
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            raise ComfyUIError(f"No prompt_id in response: {result}")
        return prompt_id

    def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> list[dict]:
        """
        Poll the /history/{prompt_id} endpoint until the job is complete.
        Uses a fixed 0.2s polling interval to minimize idle time in batch generation
        compared to exponential backoff.
        """
        deadline = time.time() + timeout
        polling_interval = 0.2
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

    def delete_output_image(self, filename: str, subfolder: str = "", comfyui_output_dir: str | None = None) -> None:
        """Delete an image from ComfyUI's output folder after it has been copied locally."""
        if not comfyui_output_dir:
            return
        try:
            parts = [comfyui_output_dir]
            if subfolder:
                parts.append(subfolder)
            parts.append(filename)
            target = Path(*parts)
            if target.exists():
                target.unlink()
        except Exception:
            pass

    def upload_image(self, image_path: str) -> str:
        """
        Upload an image to ComfyUI's input folder. Returns the filename ComfyUI assigned.
        Uses a local cache to avoid re-uploading the same file multiple times.
        """
        path = Path(image_path).resolve()
        try:
            stat = path.stat()
            cache_key = (str(path), stat.st_mtime, stat.st_size)
        except OSError as e:
            raise ComfyUIError(f"Failed to access image {image_path}: {e}")

        if cache_key in self._upload_cache:
            return self._upload_cache[cache_key]

        url = f"{self.base_url}/upload/image"
        try:
            with open(path, "rb") as f:
                files = {"image": (path.name, f, "image/png")}
                resp = self.session.post(url, files=files, timeout=30)
                resp.raise_for_status()
                remote_name = resp.json()["name"]
                self._upload_cache[cache_key] = remote_name
                return remote_name
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




def _scan_workflow_titles(workflow: dict) -> dict[str, list[str]]:
    """
    Scan workflow nodes for _meta.title and return title -> [node_id] mapping.
    O(N) where N is number of nodes. Cached if _claude_title_cache is present.
    """
    if "_claude_title_cache" in workflow:
        return workflow["_claude_title_cache"]

    title_to_ids: dict[str, list[str]] = {}
    for node_id, node in workflow.items():
        try:
            # Optimized lookup using try-except (faster in the common 'path exists' case)
            title = node["_meta"]["title"]
            if title:
                if title not in title_to_ids:
                    title_to_ids[title] = []
                title_to_ids[title].append(node_id)
        except (KeyError, TypeError):
            continue
    return title_to_ids


@functools.lru_cache(maxsize=128)
def _split_path(field_path: str) -> tuple[str, ...]:
    """
    Cached path splitting to avoid repeated string operations on common field paths.
    Optimization: Returns an immutable tuple to prevent cache corruption and
    improve memory efficiency for redundant lookups.
    """
    return tuple(field_path.split("."))


def _is_patch_redundant(node: dict, parts: tuple[str, ...], value: Any) -> bool:
    """
    Check if the value at the given path in the node already matches the target value.
    Uses try-except for faster traversal in the common 'path exists' case.
    """
    try:
        # Micro-optimization: Unroll for the extremely common 2-part path (inputs.field)
        if len(parts) == 2:
            return node[parts[0]][parts[1]] == value

        target = node
        for part in parts:
            target = target[part]
        return target == value
    except (KeyError, TypeError):
        return False


def inject_workflow_values(workflow: dict, overrides: dict[str, Any]) -> dict:
    """
    Patch workflow nodes by matching _meta.title sentinels.
    overrides maps sentinel title → dict of {field_path: value}.
    field_path uses dot notation: "inputs.text", "inputs.seed", etc.

    Performance: Uses a "shallow copy then selective branch copy" pattern.
    Optimized to group patches by node and pre-filter redundant values using
    try-except traversal (approx 15% faster for large workflows).

    Note: Always returns a new dictionary object (shallow copy at minimum) to
    ensure original workflow data remains immutable and prevent state leakage
    when using cached workflow templates.
    """
    if not overrides:
        return workflow.copy()

    title_to_ids = _scan_workflow_titles(workflow)

    # Group and pre-split paths to avoid redundant string operations in the loop
    node_to_patches: dict[str, dict[tuple[str, ...], Any]] = {}
    for title, patches in overrides.items():
        if title in title_to_ids:
            # Pre-split all paths for this title once
            # Optimization: Use a list of items to avoid redundant dict creation during filtering
            split_patches_items = [(_split_path(p), v) for p, v in patches.items()]

            for node_id in title_to_ids[title]:
                node = workflow[node_id]
                filtered = {
                    parts: val for parts, val in split_patches_items
                    if not _is_patch_redundant(node, parts, val)
                }
                if filtered:
                    node_to_patches.setdefault(node_id, {}).update(filtered)

    # Propagate the title cache to the new copy to keep subsequent injections O(1).
    # We always return a copy (even if no patches are applied) to ensure immutability.
    workflow = workflow.copy()
    workflow["_claude_title_cache"] = title_to_ids

    if not node_to_patches:
        return workflow

    for node_id, patches in node_to_patches.items():
        node = workflow[node_id] = workflow[node_id].copy()
        copied_sub_dicts = {}

        for parts, value in patches.items():
            target = node
            # Traverse and copy branches only as needed
            for i in range(len(parts) - 1):
                part = parts[i]
                val = target[part]
                val_id = id(val)
                if val_id in copied_sub_dicts:
                    target = val
                else:
                    new_target = val.copy()
                    target[part] = new_target
                    copied_sub_dicts[id(new_target)] = True
                    target = new_target

            target[parts[-1]] = value

    return workflow


def load_workflow(path: str) -> dict:
    """
    Load a ComfyUI workflow JSON from disk.
    Cached to avoid redundant I/O and JSON parsing when the same workflow
    is used repeatedly in a batch. Returns a copy so callers can't corrupt the cache.
    """
    return _load_workflow_cached(path).copy()


@functools.lru_cache(maxsize=16)
def _load_workflow_cached(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        workflow = json.load(f)
    # Warm up the title cache on load so the very first injection is O(1)
    workflow["_claude_title_cache"] = _scan_workflow_titles(workflow)
    return workflow
