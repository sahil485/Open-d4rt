"""ScanNet++ DSLR from the spatial-rep-learning indexed container (local HDF5 shards).

Each DSLR record is the ordered sequence of still photos of one scene (capture order is the
time axis), with laser-scan-frame COLMAP poses, a fisheye camera model and mesh-rendered
metric depth per frame. Frames and depth are undistorted to the pinhole model whose
normalized intrinsics the container stores, resized to the training size, and turned into
D4RT queries with the shared static-scene builder (``build_queries_from_depth``), exactly
like the dense iPhone RGB-D path of ``scannet_raw_dataset``.

Requires the ``spatial_rep`` package (streaming reader) on ``PYTHONPATH``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .bad_sample_registry import (
    BadSampleRegistry,
    RetryableSampleError,
    failed_paths_from_exception,
    is_retryable_data_error,
)
from .depth_query_builder import build_queries_from_depth
from .raw_augment import (
    RawAugmentConfig,
    apply_photometric_augment,
    apply_spatial_crop_images_only,
    build_augment_info,
    sample_frame_indices_with_stride,
)
from .seeding import SeededDatasetMixin


@dataclass
class ScannetppPackedConfig:
    root: Path  # split directory holding hdf5/, index/, manifest.json
    clip_frames: int
    image_size: tuple[int, int]  # (H, W)
    queries_per_clip: int
    hard_query_ratio: float
    prob_t_tgt_equals_t_cam: float
    training: bool
    t_src_tgt_delta_choices: tuple[int | None, ...] | None = None
    t_src_tgt_delta_probs: tuple[float, ...] | None = None
    device: str = "dslr"  # record suffix to keep ("dslr" or "iphone")
    view_splits: tuple[int, ...] = (0,)  # container split flags to keep (0 = train views; 2 = held-out)
    samples_per_scene: int = 20  # training epoch length multiplier
    max_scenes: int | None = None
    augment: RawAugmentConfig | None = None
    bad_sample_registry_path: Path = Path("data/meta/bad_sample.json")
    max_sample_retries: int = 64


@dataclass
class _Scene:
    k: int  # scene index in the SceneIndex
    scene_id: str
    views: np.ndarray  # absolute view indices kept, in capture order
    src_h: int
    src_w: int


class ScannetppPackedDataset(SeededDatasetMixin, Dataset):
    """Dense depth supervision from packed ScanNet++ DSLR records."""

    def __init__(self, config: ScannetppPackedConfig) -> None:
        from spatial_rep.data.streaming import LocalByteSource, SceneIndex

        self.cfg = config
        self.h, self.w = config.image_size
        self._init_dataset_seeding(namespace="scannetpp_packed", default_seed=20261025)
        self.augment = config.augment or RawAugmentConfig()
        if not config.training:
            self.augment = RawAugmentConfig()
        self.bad_registry = BadSampleRegistry(path=config.bad_sample_registry_path)
        self.max_sample_retries = max(1, int(config.max_sample_retries))

        root = Path(config.root)
        if not (root / "manifest.json").is_file():
            raise FileNotFoundError(f"ScanNet++ packed split not found (no manifest.json): {root}")
        self.index = SceneIndex(LocalByteSource(root))
        suffix = "/" + str(config.device)
        keep = set(int(v) for v in config.view_splits)
        self.scenes: list[_Scene] = []
        for k in range(len(self.index)):
            sid = self.index.scene_id(k)
            if not sid.endswith(suffix):
                continue
            i, s = self.index._scene_location(k)
            sh = self.index.shards[i]
            v0, v1 = sh.views_of(s)
            views = np.arange(v0, v1, dtype=np.int64)
            views = views[np.isin(sh.split[v0:v1].astype(int), list(keep))]
            if views.size < int(config.clip_frames):
                continue
            src_h, src_w = sh.sizes[s]
            self.scenes.append(_Scene(k=k, scene_id=sid, views=views, src_h=int(src_h), src_w=int(src_w)))
            if config.max_scenes is not None and len(self.scenes) >= int(config.max_scenes):
                break
        if not self.scenes:
            raise ValueError(f"No {config.device} scene with >= {config.clip_frames} kept views under {root}")
        self._grids: dict[tuple, torch.Tensor] = {}

    # ------------------------------------------------------------------ sampling
    def __len__(self) -> int:
        n = len(self.scenes)
        return n * int(self.cfg.samples_per_scene) if self.cfg.training else n

    def _scene(self, index: int) -> _Scene:
        if self.cfg.training:
            return self.scenes[int(self.rng.integers(0, len(self.scenes)))]
        return self.scenes[index % len(self.scenes)]

    def _start(self, scene: _Scene, index: int) -> int:
        max_start = int(scene.views.size) - int(self.cfg.clip_frames)
        if max_start <= 0:
            return 0
        if self.cfg.training:
            return int(self.rng.integers(0, max_start + 1))
        return int((index * self.cfg.clip_frames) % (max_start + 1))

    def _frame_indices(self, scene_len: int, index: int) -> list[int]:
        return sample_frame_indices_with_stride(
            rng=self.rng,
            scene_len=scene_len,
            clip_frames=int(self.cfg.clip_frames),
            cfg=self.augment,
            training=bool(self.cfg.training),
            index=index,
        )

    # ------------------------------------------------------------------ geometry helpers
    def _grid(self, height: int, width: int, intr: np.ndarray, cam: dict) -> torch.Tensor | None:
        """Pinhole->fisheye sampling grid for one view (None for pinhole cameras)."""
        from spatial_rep.data.streaming.hf_scenes import fisheye_grid

        model = cam.get("model", "PINHOLE")
        if model in ("PINHOLE", "SIMPLE_PINHOLE"):
            return None
        if model != "OPENCV_FISHEYE" or len(cam.get("params", [])) != 8:
            raise ValueError(f"unsupported camera model {model}")
        k = tuple(float(v) for v in cam["params"][4:8])
        key = (height, width, tuple(np.round(np.asarray(intr).reshape(-1), 7).tolist()), k)
        grid = self._grids.get(key)
        if grid is None:
            grid = self._grids[key] = fisheye_grid(height, width, intr, k)
        return grid

    def _undistort(
        self, image: np.ndarray, depth: np.ndarray, valid: np.ndarray, grid: torch.Tensor | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resample fisheye RGB (bilinear), depth and mask (nearest) onto the pinhole image plane."""
        if grid is None:
            return image, depth, valid
        img = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).float()
        img = torch.nn.functional.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
        d = torch.from_numpy(np.ascontiguousarray(depth, dtype=np.float32))[None, None]
        m = torch.from_numpy(np.ascontiguousarray(valid, dtype=np.float32))[None, None]
        dm = torch.cat([d, m], dim=1)
        dm = torch.nn.functional.grid_sample(dm, grid, mode="nearest", padding_mode="zeros", align_corners=True)[0]
        out_img = img.permute(1, 2, 0).clamp_(0, 255).round_().to(torch.uint8).numpy()
        out_d = dm[0].numpy()
        out_m = (dm[1].numpy() > 0.5) & (out_d > 0.0)
        out_d = np.where(out_m, out_d, 0.0).astype(np.float32)
        return out_img, out_d, out_m

    # ------------------------------------------------------------------ sample
    def _getitem(self, scene: _Scene, start: int, idxs: list[int]) -> dict[str, Any]:
        views = scene.views[np.asarray(idxs, dtype=np.int64)]
        order = np.argsort(views)  # fetch_views returns unique sorted views
        try:
            rec = self.index.fetch_views(scene.k, views, fetch_depth=True)
        except Exception as exc:
            raise RetryableSampleError(
                f"fetch_views failed: scene={scene.scene_id} views={views.tolist()}: {exc}",
                failed_paths=[str(self.cfg.root / "hdf5" / f"{self.index.shards[self.index._scene_location(scene.k)[0]].shard}.h5")],
            ) from exc
        if rec.depth is None or rec.depth_mask is None:
            raise ValueError(f"{scene.scene_id}: record has no depth")
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)  # position of the j-th requested view in rec arrays
        pos = inv  # requested order -> rec index

        t = len(idxs)
        frames: list[np.ndarray] = []
        depth_list: list[np.ndarray] = []
        valid_list: list[np.ndarray] = []
        k_seq = np.zeros((t, 3, 3), dtype=np.float32)
        t_wc = np.zeros((t, 4, 4), dtype=np.float32)
        for j in range(t):
            r = int(pos[j])
            intr = rec.intr[r]
            cam = rec.cameras.get(str(int(rec.camera_id[r])), {})
            grid = self._grid(scene.src_h, scene.src_w, intr, cam)
            depth = rec.depth[r].astype(np.float32)
            valid = rec.depth_mask[r].astype(bool) & np.isfinite(depth) & (depth > 0.0)
            image, depth, valid = self._undistort(rec.images[r], depth, valid, grid)
            if (scene.src_h, scene.src_w) != (self.h, self.w):
                image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_AREA)
                depth = cv2.resize(depth, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
                valid = cv2.resize(valid.astype(np.uint8), (self.w, self.h), interpolation=cv2.INTER_NEAREST).astype(bool)
            frames.append(image)
            depth_list.append(np.where(valid, depth, 0.0).astype(np.float32))
            valid_list.append(valid)
            # normalized intrinsics -> pixels of the output size (pinhole after undistortion)
            k_seq[j] = np.array(
                [[intr[0, 0] * self.w, 0.0, intr[0, 2] * self.w], [0.0, intr[1, 1] * self.h, intr[1, 2] * self.h], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            t_wc[j] = rec.c2w[r]  # OpenCV camera-to-world in the metric laser-scan frame

        video = np.transpose(np.stack(frames, axis=0).astype(np.float32) / 255.0, (0, 3, 1, 2))
        depth_arr = np.stack(depth_list, axis=0)
        depth_valid = np.stack(valid_list, axis=0)
        aspect_ratio = np.array([scene.src_w / max(1.0, scene.src_h)], dtype=np.float32)
        cam_valid = np.ones((t,), dtype=np.bool_)
        _crop_info: dict = {}
        if self.cfg.training:
            video = apply_photometric_augment(video, self.rng, self.augment)
            (video, depth_arr, depth_valid, k_seq, aspect_ratio) = apply_spatial_crop_images_only(
                video_t_chw=video,
                depth_t_hw=depth_arr,
                depth_valid_t_hw=depth_valid,
                k_t_33=k_seq,
                camera_valid_t=cam_valid,
                rng=self.rng,
                cfg=self.augment,
                native_aspect_ratio=aspect_ratio,
                out_info=_crop_info,
            )

        query, target, mask, query_stats = build_queries_from_depth(
            rng=self.rng,
            depth=depth_arr,
            depth_valid=depth_valid,
            k_seq=k_seq,
            t_wc_seq=t_wc,
            camera_valid=cam_valid,
            queries_per_clip=int(self.cfg.queries_per_clip),
            hard_query_ratio=float(self.cfg.hard_query_ratio),
            prob_t_tgt_equals_t_cam=float(self.cfg.prob_t_tgt_equals_t_cam),
            t_src_tgt_delta_choices=self.cfg.t_src_tgt_delta_choices,
            t_src_tgt_delta_probs=self.cfg.t_src_tgt_delta_probs,
        )
        return {
            "video": torch.from_numpy(video).float(),
            "aspect_ratio": torch.from_numpy(aspect_ratio.astype(np.float32)),
            "depth_m": torch.from_numpy(depth_arr).float(),
            "depth_valid": torch.from_numpy(depth_valid).bool(),
            "query": {k: torch.from_numpy(v).to(torch.long if k.startswith("t_") else torch.float32) for k, v in query.items()},
            "query_stats": {k: torch.from_numpy(v).bool() for k, v in query_stats.items()},
            "target": {k: torch.from_numpy(v).float() for k, v in target.items()},
            "mask": {k: torch.from_numpy(v).bool() for k, v in mask.items()},
            "camera": {
                "K": torch.from_numpy(k_seq).float(),
                "T_wc": torch.from_numpy(t_wc).float(),
                "camera_valid": torch.from_numpy(cam_valid).bool(),
            },
            "augment_info": {k: torch.from_numpy(v) for k, v in build_augment_info(_crop_info, image_hw=(self.h, self.w)).items()},
            "meta": {
                "dataset": "scannetpp_packed",
                "scene_id": scene.scene_id,
                "clip_start": start,
                "source_mode": f"{self.cfg.device}_mesh_depth",
                "native_hw": (scene.src_h, scene.src_w),
            },
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        last_error: Exception | None = None
        total = max(1, len(self))
        for attempt in range(self.max_sample_retries):
            query_index, _ = self._prepare_sample_rng(index=index, total=total, attempt=attempt)
            scene = self._scene(query_index)
            start = self._start(scene, query_index)
            idxs = self._frame_indices(int(scene.views.size), start)
            sample_key = f"scannetpp_packed::{scene.scene_id}::views={','.join(str(int(scene.views[i])) for i in idxs)}"
            if self.bad_registry.is_bad_sample(sample_key):
                continue
            try:
                sample = self._getitem(scene, start, idxs)
            except Exception as exc:
                if not is_retryable_data_error(exc):
                    raise
                last_error = exc
                self.bad_registry.mark_bad(
                    dataset="scannetpp_packed",
                    sample_key=sample_key,
                    sample_paths=[],
                    failed_paths=failed_paths_from_exception(exc),
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue
            sample["meta"]["sample_key"] = sample_key
            return sample
        raise RuntimeError(
            f"ScannetppPackedDataset failed to produce a valid sample after {self.max_sample_retries} retries. last_error={last_error}"
        )
