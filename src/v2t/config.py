"""Configuration for Video-to-Terrain MVP pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env from project root
# Path: src/v2t/config.py -> 3 levels up = Videos_to_terrain
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Paths:
    """Output directory paths for a single run."""
    run_dir: Path

    @property
    def frames(self) -> Path:
        return self.run_dir / "frames"

    @property
    def depth(self) -> Path:
        return self.run_dir / "geometry" / "depth"

    @property
    def geometry(self) -> Path:
        return self.run_dir / "geometry"

    @property
    def holi_spatial(self) -> Path:
        return self.run_dir / "holi_spatial"

    @property
    def sam_masks(self) -> Path:
        return self.holi_spatial / "sam_masks"

    @property
    def scene(self) -> Path:
        return self.run_dir / "scene"

    @property
    def simulation(self) -> Path:
        return self.run_dir / "simulation"

    @property
    def visual(self) -> Path:
        return self.simulation / "visual"

    @property
    def collision(self) -> Path:
        return self.simulation / "collision"

    @property
    def mujoco(self) -> Path:
        return self.run_dir / "mujoco"

    @property
    def mjlab(self) -> Path:
        return self.run_dir / "mjlab"

    @property
    def validation(self) -> Path:
        return self.run_dir / "validation"


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    # Input
    input_video: Path = PROJECT_ROOT.parent / "input" / "20260810_150244_color.mp4"

    # Run output
    run_name: str = "run_mvp"
    runs_root: Path = PROJECT_ROOT / "outputs"

    @property
    def run_dir(self) -> Path:
        return self.runs_root / self.run_name

    @property
    def paths(self) -> Paths:
        return Paths(self.run_dir)

    # Frame extraction
    frame_sample_rate: int = 5  # extract every N frames
    max_frames: int = 130  # covers the full ~556-frame/39s clip at sample_rate=5 (112 samples)
    # so a full 360deg walk-around isn't truncated at the tail (was 100, cutting last ~4.2s)

    # DA3
    da3_model: str = "da3-large"  # non-metric with camera decoder for poses
    da3_process_res: int = 504
    da3_device: str = "mps"  # cuda, mps, cpu

    # VLM (NVIDIA NIM)
    vlm_provider: str = "nvidia"
    nvidia_base_url: str = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    nvidia_vlm_model: str = os.getenv("NVIDIA_VLM_MODEL", "nvidia/llama-3.1-nemotron-nano-8b-v1")
    nvidia_api_keys: list = field(default_factory=lambda: [
        os.getenv("NVIDIA_API_KEY_1", ""),
        os.getenv("NVIDIA_API_KEY_2", ""),
        os.getenv("NVIDIA_API_KEY_3", ""),
    ])

    # SAM3
    sam3_checkpoint: str | None = None  # auto-download from HF
    sam3_confidence_threshold: float = 0.6

    # Scene
    floor_label: str = "floor"
    up_axis: str = "z"

    # Physics defaults
    physics_profiles: dict = field(default_factory=lambda: {
        "floor":      {"friction": 0.8, "restitution": 0.0, "static": True,  "source": "default_profile:indoor_floor"},
        "wall":       {"friction": 0.6, "restitution": 0.0, "static": True,  "source": "default_profile:wall"},
        "table":      {"friction": 0.5, "restitution": 0.0, "static": True,  "source": "default_profile:wood_furniture"},
        "chair":      {"friction": 0.5, "restitution": 0.0, "static": False, "mass": 5.0, "source": "default_profile:wood_furniture"},
        "stairs":     {"friction": 0.7, "restitution": 0.0, "static": True,  "source": "default_profile:stairs"},
        "default":    {"friction": 0.5, "restitution": 0.1, "static": True,  "source": "default_profile:generic"},
    })

    # Robot
    robot_name: str = "pelvis"
    robot_spawn_height: float = 1.0

    # Validation
    sim_steps: int = 300

    def validate(self) -> bool:
        """Check required configs."""
        if not self.input_video.exists():
            raise FileNotFoundError(f"Input video not found: {self.input_video}")
        if all(not k or k == "nvapi-REPLACE_ME_1" for k in self.nvidia_api_keys):
            print("WARNING: No valid NVIDIA API keys found in .env. VLM will fail.")
        return True


# Global config instance
CFG = PipelineConfig()
