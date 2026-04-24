"""ComfyUI REST API client library."""

import json
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path


class ComfyUIError(Exception):
    pass


class ComfyUIClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8188):
        self.base_url = f"http://{host}:{port}"

    def _get(self, path: str) -> dict:
        url = self.base_url + path
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read())

    def _post(self, path: str, payload: dict) -> dict:
        url = self.base_url + path
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def wait_for_ready(self, timeout: int = 90) -> None:
        """Poll /system_stats with exponential backoff until ComfyUI responds."""
        deadline = time.time() + timeout
        delay = 1.0
        while time.time() < deadline:
            try:
                self._get("/system_stats")
                return
            except Exception:
                time.sleep(delay)
                delay = min(delay * 2, 10)
        raise ComfyUIError(
            f"ComfyUI not reachable at {self.base_url} after {timeout}s. "
            "Start ComfyUI and try again."
        )

    def submit_workflow(self, workflow: dict) -> str:
        """POST workflow to /prompt; return prompt_id."""
        resp = self._post("/prompt", {"prompt": workflow})
        prompt_id = resp.get("prompt_id")
        if not prompt_id:
            raise ComfyUIError(f"No prompt_id in response: {resp}")
        return prompt_id

    def wait_for_completion(self, prompt_id: str, timeout: int = 600) -> dict:
        """Poll /history/{prompt_id} until the job is done; return outputs dict."""
        deadline = time.time() + timeout
        delay = 2.0
        while time.time() < deadline:
            try:
                history = self._get(f"/history/{prompt_id}")
                if prompt_id in history:
                    entry = history[prompt_id]
                    status = entry.get("status", {})
                    if status.get("completed"):
                        return entry.get("outputs", {})
                    if status.get("status_str") == "error":
                        msgs = [m.get("text", "") for m in status.get("messages", [])]
                        raise ComfyUIError(f"ComfyUI workflow error: {' | '.join(msgs)}")
            except ComfyUIError:
                raise
            except Exception:
                pass
            time.sleep(delay)
            delay = min(delay * 1.5, 15)
        raise ComfyUIError(f"Workflow {prompt_id} did not complete within {timeout}s")

    def download_outputs(self, outputs: dict, dest_dir: str) -> list[Path]:
        """Fetch all SaveImage outputs from /view and write to dest_dir."""
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        saved = []
        for node_outputs in outputs.values():
            for images in node_outputs.values():
                if not isinstance(images, list):
                    continue
                for img in images:
                    if not isinstance(img, dict) or img.get("type") != "output":
                        continue
                    params = urllib.parse.urlencode({
                        "filename": img["filename"],
                        "subfolder": img.get("subfolder", ""),
                        "type": img["type"],
                    })
                    url = f"{self.base_url}/view?{params}"
                    out_path = dest / img["filename"]
                    with urllib.request.urlopen(url, timeout=60) as resp:
                        out_path.write_bytes(resp.read())
                    saved.append(out_path)
        return saved

    def submit_and_wait(self, workflow: dict, dest_dir: str, timeout: int = 600) -> list[Path]:
        """Submit workflow, wait for completion, download outputs. Returns saved paths."""
        prompt_id = self.submit_workflow(workflow)
        outputs = self.wait_for_completion(prompt_id, timeout=timeout)
        return self.download_outputs(outputs, dest_dir)
