import os
import time
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock MCP before importing mcp_server
sys.modules["mcp"] = MagicMock()
sys.modules["mcp.server"] = MagicMock()
sys.modules["mcp.server.fastmcp"] = MagicMock()

import scripts.mcp_server as mcp_server

def setup_test_data(root: Path, days=10, imgs_per_day=20, carousels_per_day=2):
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    for i in range(days):
        date_str = f"2026-05-{10-i:02d}"
        date_dir = root / date_str / "ananya"
        date_dir.mkdir(parents=True)

        # Create images
        for j in range(imgs_per_day):
            (date_dir / f"ananya_{date_str}_120000_{j:04d}.png").touch()

        # Create carousels
        for j in range(carousels_per_day):
            c_dir = date_dir / f"carousel_test_{j:02d}"
            c_dir.mkdir()
            for k in range(5):
                (c_dir / f"slide_{k:02d}.png").touch()
            (c_dir / "carousel_info.txt").write_text("info")

def benchmark_find_recent_images(output_dir, limit=50):
    start = time.perf_counter()
    res = mcp_server._find_recent_images(output_dir, "ananya", "ananya", limit)
    end = time.perf_counter()
    return end - start, len(res)

def benchmark_find_all_carousels(output_dir):
    start = time.perf_counter()
    res = mcp_server._find_all_carousels(output_dir, "ananya")
    end = time.perf_counter()
    return end - start, len(res)

def benchmark_report(output_dir):
    # Ensure mcp_server.ROOT is absolute for relative_to check
    mcp_server.ROOT = Path(__file__).parent.parent.resolve()
    abs_output_dir = output_dir.resolve()

    # Mock config to point to our temp output
    mcp_server._load_config = MagicMock(return_value={
        "paths": {"output_dir": str(abs_output_dir.relative_to(mcp_server.ROOT))},
        "characters": {"ananya": {"output_subdir": "ananya"}}
    })

    # Manually run report logic since FastMCP mock makes it hard to call directly as a tool
    start = time.perf_counter()
    carousel_dirs = mcp_server._find_all_carousels(abs_output_dir, "ananya")
    rows = []
    for d in carousel_dirs:
        try:
            with os.scandir(d) as it:
                slide_count = sum(1 for entry in it if entry.is_file() and entry.name.startswith("slide_") and entry.name.endswith(".png"))
        except OSError:
            slide_count = 0
        rows.append((d.name, slide_count))
    end = time.perf_counter()
    return end - start, len(rows)

if __name__ == "__main__":
    temp_root = Path("temp_benchmark_output").resolve()
    print(f"Setting up test data in {temp_root}...")
    setup_test_data(temp_root, days=10, imgs_per_day=20, carousels_per_day=2)

    print("\nBenchmarking _find_recent_images (limit 50):")
    t, n = benchmark_find_recent_images(temp_root, limit=50)
    print(f"Time: {t:.6f}s, Found: {n}")

    print("\nBenchmarking _find_all_carousels:")
    t, n = benchmark_find_all_carousels(temp_root)
    print(f"Time: {t:.6f}s, Found: {n}")

    print("\nBenchmarking content_readiness_report logic:")
    t, n = benchmark_report(temp_root)
    print(f"Time: {t:.6f}s, Processed: {n} carousels")

    shutil.rmtree(temp_root)
