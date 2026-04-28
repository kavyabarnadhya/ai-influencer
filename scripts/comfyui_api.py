import json
import time
import random
import urllib.parse
import urllib.request
import urllib.error
from typing import Any


class ComfyUIError(Exception):
    pass


class ComfyUIClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8188):
        self.base_url = f"http://{host}:{port}"
        self.client_id = str(random.randint(100000, 999999))

    def _get(self, path: str) -> Any:
        url = f"{self.base_url}{path}"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as e:
            raise ComfyUIError(f"ComfyUI unreachable at {self.base_url}: {e}")

    def _post(self, path: str, data: dict) -> Any:
        url = f"{self.base_url}{path}"
        payload = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise ComfyUIError(f"POST to {path} failed: {e}\nResponse: {error_body}")
        except urllib.error.URLError as e:
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
        deadline = time.time() + timeout
        delay = 2.0
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
            time.sleep(delay)
            delay = min(delay * 1.3, 15.0)
        raise ComfyUIError(f"Prompt {prompt_id} did not complete within {timeout}s")

    def download_image(self, filename: str, subfolder: str = "", img_type: str = "output") -> bytes:
        params = f"filename={urllib.parse.quote(filename)}&subfolder={urllib.parse.quote(subfolder)}&type={img_type}"
        url = f"{self.base_url}/view?{params}"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read()
        except urllib.error.URLError as e:
            raise ComfyUIError(f"Failed to download image {filename}: {e}")

    def upload_image(self, image_path: str) -> str:
        """Upload an image to ComfyUI's input folder. Returns the filename ComfyUI assigned."""
        from pathlib import Path
        path = Path(image_path)
        boundary = f"ComfyBoundary{random.randint(100000, 999999)}"
        with open(path, "rb") as f:
            file_data = f.read()
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
            f"Content-Type: image/png\r\n"
            f"\r\n"
        ).encode("utf-8") + file_data + f"\r\n--{boundary}--\r\n".encode("utf-8")
        url = f"{self.base_url}/upload/image"
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                return result["name"]
        except urllib.error.URLError as e:
            raise ComfyUIError(f"Failed to upload image {path.name}: {e}")


# Global cache for workflow title-to-ID mappings to speed up batch injections
_WORKFLOW_TITLE_CACHE: dict[int, dict[str, list[str]]] = {}


def inject_workflow_values(workflow: dict, overrides: dict[str, Any]) -> dict:
    """
    Patch workflow nodes by matching _meta.title sentinels.
    overrides maps sentinel title → dict of {field_path: value}.
    field_path uses dot notation: "inputs.text", "inputs.seed", etc.
    """
    if not overrides:
        return workflow

    # Optimization: Use a cached mapping of title -> node_ids for the workflow object.
    # This avoids O(N) scan of all nodes for every injection in a batch.
    wf_id = id(workflow)
    if wf_id not in _WORKFLOW_TITLE_CACHE:
        # Limit cache size to avoid memory leaks if many different workflows are loaded
        if len(_WORKFLOW_TITLE_CACHE) > 10:
            _WORKFLOW_TITLE_CACHE.clear()

        mapping = {}
        for node_id, node in workflow.items():
            title = node.get("_meta", {}).get("title")
            if title:
                if title not in mapping:
                    mapping[title] = []
                mapping[title].append(node_id)
        _WORKFLOW_TITLE_CACHE[wf_id] = mapping

    title_to_ids = _WORKFLOW_TITLE_CACHE[wf_id]

    # Performance Optimization: Instead of deep-copying the entire workflow,
    # we shallow copy the top-level dict and only deep-copy branches that need patching.
    workflow = workflow.copy()

    for title, patches in overrides.items():
        if title not in title_to_ids:
            continue

        for node_id in title_to_ids[title]:
            node = workflow[node_id]
            # Selective copy: To optimize for speed while maintaining correctness,
            # we only deep-copy the branches of the node's dictionary that are
            # actually being modified by the patches.
            node = node.copy()
            workflow[node_id] = node

            for field_path, value in patches.items():
                parts = field_path.split(".")
                target = node
                # Traverse and shallow-copy as we go to ensure we don't mutate original
                for part in parts[:-1]:
                    prev_target = target
                    target = target[part].copy()
                    prev_target[part] = target
                target[parts[-1]] = value

    return workflow


def load_workflow(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
