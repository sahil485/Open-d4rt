"""DyCheck iPhone (Gao et al., NeurIPS 2022) raw-release adapter: monocular dynamic RGB-D video.

Reads the official ``datasets/iphone/<scene>/`` folders: ``rgb/{factor}x/<cam>_<frame>.png``,
``depth/{factor}x/<cam>_<frame>.npy`` (LiDAR z-depth, metres), ``camera/<cam>_<frame>.json``
(Nerfies camera: ``orientation`` = world->camera rotation, ``position`` = camera centre, OpenCV
axes) and ``splits/train.json`` (the moving training camera ``0``).

Supervision comes from ``build_queries_from_depth``. That builder assumes a static world when
``t_tgt != t_src``; DyCheck scenes are dynamic, so configs for this dataset should restrict
``train_sampling.timestep_sampling.t_src_tgt_delta_choices`` to ``[0]`` (``t_tgt == t_src``),
which turns every target into the LiDAR point of frame ``t_src`` rigidly re-expressed in the
camera of frame ``t_cam``. That stays exact for moving objects; only the visibility label
ignores occluders that moved between ``t_src`` and ``t_cam``.
"""

from __future__ import annotations

import json
import warnings
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
class DycheckRawConfig:
    root: Path  # directory holding <scene>/{rgb,depth,camera,splits}
    clip_frames: int
    image_size: tuple[int, int]  # (H, W)
    queries_per_clip: int
    hard_query_ratio: float
    prob_t_tgt_equals_t_cam: float
    training: bool
    t_src_tgt_delta_choices: tuple[int | None, ...] | None = None
    t_src_tgt_delta_probs: tuple[float, ...] | None = None
    factor: int = 2  # which rgb/depth downsample folder to read (1 or 2)
    split_name: str = "train"  # splits/<split_name>.json: which frames/cameras form the sequence
    scenes: tuple[str, ...] | None = None  # explicit scene list (default: every scene dir with splits/)
    exclude_scenes: tuple[str, ...] = ()
    samples_per_scene: int = 200  # training epoch length multiplier
    max_depth_m: float = 0.0  # drop LiDAR depth beyond this (0 = keep all finite > 0)
    augment: RawAugmentConfig | None = None
    bad_sample_registry_path: Path = Path("data/meta/bad_sample.json")
    max_sample_retries: int = 64


@dataclass
class _Frame:
    name: str  # "<cam>_<frame>" stem shared by rgb/depth/camera
    k: np.ndarray  # 3x3 pixel intrinsics at the *loaded* (factor) resolution
    t_wc: np.ndarray  # 4x4 camera-to-world, OpenCV axes, metres


@dataclass
class _Scene:
    name: str
    path: Path
    frames: list[_Frame]  # time-ordered
    src_h: int
    src_w: int


def _load_camera(path: Path, factor: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Nerfies/DyCheck camera json -> (K at 1/factor scale, T_wc, (H, W) at 1/factor scale)."""
    cam = json.loads(path.read_text())
    r_cw = np.asarray(cam["orientation"], dtype=np.float64).reshape(3, 3)  # world -> camera
    pos = np.asarray(cam["position"], dtype=np.float64).reshape(3)
    t_wc = np.eye(4, dtype=np.float64)
    t_wc[:3, :3] = r_cw.T
    t_wc[:3, 3] = pos
    f = float(cam["focal_length"])
    par = float(cam.get("pixel_aspect_ratio", 1.0))
    skew = float(cam.get("skew", 0.0))
    cx, cy = (float(v) for v in cam["principal_point"])
    w_full, h_full = (int(v) for v in cam["image_size"])
    radial = np.asarray(cam.get("radial_distortion", [0.0, 0.0, 0.0]), dtype=np.float64)
    tangential = np.asarray(cam.get("tangential_distortion", [0.0, 0.0]), dtype=np.float64)
    if np.abs(radial).max() > 1e-6 or np.abs(tangential).max() > 1e-6:
        raise ValueError(f"{path}: distortion is not supported (radial={radial.tolist()}, tangential={tangential.tolist()})")
    s = 1.0 / float(factor)
    k = np.array([[f * s, skew * s, cx * s], [0.0, f * par * s, cy * s], [0.0, 0.0, 1.0]], dtype=np.float64)
    return k, t_wc, (h_full // factor, w_full // factor)


class DycheckRawDataset(SeededDatasetMixin, Dataset):
    """Dense LiDAR-depth supervision from DyCheck iPhone monocular videos."""

    def __init__(self, config: DycheckRawConfig) -> None:
        self.cfg = config
        self.h, self.w = config.image_size
        self._init_dataset_seeding(namespace="dycheck_raw", default_seed=20261025)
        self.augment = config.augment or RawAugmentConfig()
        if not config.training:
            self.augment = RawAugmentConfig()
        self.bad_registry = BadSampleRegistry(path=config.bad_sample_registry_path)
        self.max_sample_retries = max(1, int(config.max_sample_retries))

        root = Path(config.root)
        if config.scenes:
            names = [str(s) for s in config.scenes]
        else:
            names = sorted(p.name for p in root.iterdir() if (p / "splits" / f"{config.split_name}.json").is_file())
        names = [n for n in names if n not in set(config.exclude_scenes)]
        if not names:
            raise FileNotFoundError(f"No DyCheck scenes with splits/{config.split_name}.json under {root}")

        self.scenes: list[_Scene] = []
        for name in names:
            scene = self._index_scene(root / name)
            if scene is None:
                continue
            if len(scene.frames) < int(config.clip_frames):
                warnings.warn(f"dycheck: skipping {name}: {len(scene.frames)} frames < clip_frames={config.clip_frames}")
                continue
            self.scenes.append(scene)
        if not self.scenes:
            raise ValueError(f"No DyCheck scene with >= {config.clip_frames} frames under {root}")

    # ------------------------------------------------------------------ indexing
    def _index_scene(self, path: Path) -> _Scene | None:
        split = json.loads((path / "splits" / f"{self.cfg.split_name}.json").read_text())
        frame_names = [str(n) for n in split["frame_names"]]
        time_ids = [int(t) for t in split.get("time_ids", range(len(frame_names)))]
        order = np.argsort(np.asarray(time_ids, dtype=np.int64), kind="stable")
        frames: list[_Frame] = []
        hw: tuple[int, int] | None = None
        for i in order:
            fname = frame_names[int(i)]
            cam_path = path / "camera" / f"{fname}.json"
            rgb_path = path / "rgb" / f"{self.cfg.factor}x" / f"{fname}.png"
            dep_path = path / "depth" / f"{self.cfg.factor}x" / f"{fname}.npy"
            if not (cam_path.is_file() and rgb_path.is_file() and dep_path.is_file()):
                warnings.warn(f"dycheck: {path.name}/{fname}: missing rgb/depth/camera file; skipping frame")
                continue
            k, t_wc, cur_hw = _load_camera(cam_path, int(self.cfg.factor))
            if hw is None:
                hw = cur_hw
            elif cur_hw != hw:
                raise ValueError(f"{path.name}: inconsistent image sizes {hw} vs {cur_hw}")
            frames.append(_Frame(name=fname, k=k.astype(np.float32), t_wc=t_wc.astype(np.float32)))
        if not frames or hw is None:
            return None
        return _Scene(name=path.name, path=path, frames=frames, src_h=int(hw[0]), src_w=int(hw[1]))

    # ------------------------------------------------------------------ sampling
    def __len__(self) -> int:
        n = len(self.scenes)
        return n * int(self.cfg.samples_per_scene) if self.cfg.training else n * 4

    def _scene(self, index: int) -> _Scene:
        if self.cfg.training:
            return self.scenes[int(self.rng.integers(0, len(self.scenes)))]
        return self.scenes[index % len(self.scenes)]

    def _start(self, scene: _Scene, index: int) -> int:
        max_start = len(scene.frames) - int(self.cfg.clip_frames)
        if max_start <= 0:
            return 0
        if self.cfg.training:
            return int(self.rng.integers(0, max_start + 1))
        return int((index // len(self.scenes)) * self.cfg.clip_frames % (max_start + 1))

    def _frame_indices(self, scene_len: int, start: int) -> list[int]:
        return sample_frame_indices_with_stride(
            rng=self.rng,
            scene_len=scene_len,
            clip_frames=int(self.cfg.clip_frames),
            cfg=self.augment,
            training=bool(self.cfg.training),
            index=start,
        )

    # ------------------------------------------------------------------ io
    def _read_frame(self, scene: _Scene, fr: _Frame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb_path = scene.path / "rgb" / f"{self.cfg.factor}x" / f"{fr.name}.png"
        dep_path = scene.path / "depth" / f"{self.cfg.factor}x" / f"{fr.name}.npy"
        try:
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise OSError("cv2.imread returned None")
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            raise RetryableSampleError(f"Failed to read RGB: {rgb_path}: {exc}", failed_paths=[str(rgb_path)]) from exc
        try:
            depth = np.load(dep_path)
        except Exception as exc:
            raise RetryableSampleError(f"Failed to read depth: {dep_path}: {exc}", failed_paths=[str(dep_path)]) from exc
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if image.shape[:2] != (scene.src_h, scene.src_w) or depth.shape != (scene.src_h, scene.src_w):
            raise ValueError(
                f"{scene.name}/{fr.name}: rgb {image.shape[:2]} depth {depth.shape} vs camera {(scene.src_h, scene.src_w)}"
            )
        valid = np.isfinite(depth) & (depth > 0.0)
        if self.cfg.max_depth_m > 0:
            valid &= depth <= float(self.cfg.max_depth_m)
        return image, np.where(valid, depth, 0.0).astype(np.float32), valid

    # ------------------------------------------------------------------ sample
    def _getitem(self, scene: _Scene, start: int, idxs: list[int]) -> dict[str, Any]:
        t = len(idxs)
        frames: list[np.ndarray] = []
        depth_list: list[np.ndarray] = []
        valid_list: list[np.ndarray] = []
        k_seq = np.zeros((t, 3, 3), dtype=np.float32)
        t_wc = np.zeros((t, 4, 4), dtype=np.float32)
        sx, sy = self.w / float(scene.src_w), self.h / float(scene.src_h)
        for j, i in enumerate(idxs):
            fr = scene.frames[int(i)]
            image, depth, valid = self._read_frame(scene, fr)
            if (scene.src_h, scene.src_w) != (self.h, self.w):
                image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_AREA)
                depth = cv2.resize(depth, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
                valid = cv2.resize(valid.astype(np.uint8), (self.w, self.h), interpolation=cv2.INTER_NEAREST).astype(bool)
            frames.append(image)
            depth_list.append(np.where(valid, depth, 0.0).astype(np.float32))
            valid_list.append(valid)
            k = fr.k.astype(np.float64).copy()
            k[0, :] *= sx
            k[1, :] *= sy
            k_seq[j] = k.astype(np.float32)
            t_wc[j] = fr.t_wc

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
                "dataset": "dycheck_raw",
                "scene_id": scene.name,
                "clip_start": start,
                "source_mode": f"iphone_lidar_depth_{self.cfg.factor}x",
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
            idxs = self._frame_indices(len(scene.frames), start)
            sample_key = f"dycheck_raw::{scene.name}::frames={','.join(scene.frames[i].name for i in idxs)}"
            if self.bad_registry.is_bad_sample(sample_key):
                continue
            try:
                sample = self._getitem(scene, start, idxs)
            except Exception as exc:
                if not is_retryable_data_error(exc):
                    raise
                last_error = exc
                self.bad_registry.mark_bad(
                    dataset="dycheck_raw",
                    sample_key=sample_key,
                    sample_paths=[],
                    failed_paths=failed_paths_from_exception(exc),
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue
            sample["meta"]["sample_key"] = sample_key
            return sample
        raise RuntimeError(
            f"DycheckRawDataset failed to produce a valid sample after {self.max_sample_retries} retries. last_error={last_error}"
        )
