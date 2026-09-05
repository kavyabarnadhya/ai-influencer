import tempfile
import time
import os
import concurrent.futures
from PIL import Image
import torchvision.transforms as T
import torch
from rich.console import Console

from clip_similarity_audit import encode_images

console = Console()

def encode_images_old(image_paths: list[str], model, preprocess, device, torch, batch_size: int = 16) -> list:
    """Sequential implementation for benchmark comparison."""
    features_list = [None] * len(image_paths)

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_imgs = []
        valid_indices = []

        for j, p in enumerate(batch_paths):
            try:
                img = preprocess(Image.open(p).convert("RGB"))
                batch_imgs.append(img)
                valid_indices.append(i + j)
            except Exception as e:
                console.print(f"  [yellow]Skip {os.path.basename(p)}: {e}[/yellow]")

        if not batch_imgs:
            continue

        try:
            imgs_tensor = torch.stack(batch_imgs).to(device)
            is_cuda = str(device).startswith("cuda")
            autocast_ctx = torch.amp.autocast("cuda" if is_cuda else "cpu", enabled=is_cuda)

            with torch.no_grad(), autocast_ctx:
                feats = model.encode_image(imgs_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                feats_cpu = feats.cpu()

            for k, idx in enumerate(valid_indices):
                features_list[idx] = feats_cpu[k].unsqueeze(0)
        except Exception as e:
            console.print(f"  [red]Batch processing error at index {i}: {e}[/red]")

    return features_list


def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_paths = []
        for i in range(32):
            img = Image.new('RGB', (1024, 1024), color=(i*7 % 256, i*13 % 256, i*17 % 256))
            p = os.path.join(tmp_dir, f"img_{i:02d}.jpg")
            img.save(p, quality=95)
            image_paths.append(p)

        # Mock model for benchmarking I/O + preprocess pipeline
        class MockModel:
            def encode_image(self, tensor):
                return torch.randn(tensor.shape[0], 512)

        model = MockModel()
        preprocess = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        device = "cpu"

        # Warmup
        _ = encode_images_old(image_paths[:2], model, preprocess, device, torch)
        _ = encode_images(image_paths[:2], model, preprocess, device, torch)

        t0 = time.perf_counter()
        res_old = encode_images_old(image_paths, model, preprocess, device, torch)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        res_new = encode_images(image_paths, model, preprocess, device, torch)
        t3 = time.perf_counter()

        print(f"Sequential I/O time (32 images): {(t1-t0)*1000:.2f} ms")
        print(f"Parallel ThreadPool time (32 images): {(t3-t2)*1000:.2f} ms")
        print(f"Speedup: {(t1-t0)/(t3-t2):.2f}x")

        assert len(res_old) == len(res_new) == 32
        for r1, r2 in zip(res_old, res_new):
            assert r1.shape == r2.shape == (1, 512)

        print("Verification passed successfully!")

if __name__ == "__main__":
    main()
