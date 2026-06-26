#!/usr/bin/env python3
"""
Scoliosis pipeline master launcher.

This script provides a Gradio-based control panel that walks the operator
through the end-to-end scoliosis reconstruction workflow.  Each step opens
the existing GUI tool for that phase, provides pre-populated path hints, and
waits for the expected artifacts before progressing to the next step.
"""

from __future__ import annotations

import html
import json
import shutil
import subprocess
import sys
import threading
import time
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


@dataclass(frozen=True)
class StepSpec:
    key: str
    label: str
    script: Path
    env: Optional[str]
    config_template: Dict[str, Any]
    instructions: List[str]
    expected_globs: List[str]
    timeout_sec: int = 600


PIPELINE_STEPS: List[StepSpec] = [
    StepSpec(
        key="vid_to_frame_far",
        label="Common › Frames (far)",
        script=MODELS_DIR / "vid_to_frame" / "vid_to_frame.py",
        env="vv_vggt",
        config_template={
            "input_dir": "{SCOL}/input",
            "output_dir": "{SCOL}/working/Common/frames",
            "frames": 45,
            "videos": ["p_env_far.mp4"],
        },
        instructions=[
            "Headless extraction; frames are written to `{SCOL}/working/Common/frames/p_env_far/`.",
        ],
        expected_globs=[
            "{SCOL}/working/Common/frames/p_env_far/*.png",
        ],
        timeout_sec=600,
    ),
    StepSpec(
        key="vid_to_frame_close",
        label="Common › Frames (near)",
        script=MODELS_DIR / "vid_to_frame" / "vid_to_frame.py",
        env="vv_vggt",
        config_template={
            "input_dir": "{SCOL}/input",
            "output_dir": "{SCOL}/working/Common/frames",
            "frames": 45,
            "videos": ["p_env_close.mp4"],
        },
        instructions=[
            "Headless extraction; frames are written to `{SCOL}/working/Common/frames/p_env_close/`.",
        ],
        expected_globs=[
            "{SCOL}/working/Common/frames/p_env_close/*.png",
        ],
        timeout_sec=600,
    ),
    StepSpec(
        key="vggt_far",
        label="Common › VGGT reconstruction (far)",
        script=MODELS_DIR / "vggt" / "vggt_far.py",
        env="vv_vggt",
        config_template={
            "frames_dir": "{SCOL}/working/Common/frames/p_env_far",
            "output_glb": "{SCOL}/working/Common/vggt_reconstruction/p_env_far.glb",
            "threshold": 50,
        },
        instructions=[
            "Runs VGGT with threshold 50 and saves `p_env_far.glb` automatically.",
        ],
        expected_globs=[
            "{SCOL}/working/Common/vggt_reconstruction/p_env_far.glb",
        ],
        timeout_sec=7200,
    ),
    StepSpec(
        key="vggt_near",
        label="Common › VGGT reconstruction (near)",
        script=MODELS_DIR / "vggt" / "vggt_near.py",
        env="vv_vggt",
        config_template={
            "frames_dir": "{SCOL}/working/Common/frames/p_env_close",
            "output_glb": "{SCOL}/working/Common/vggt_reconstruction/p_env_close.glb",
            "threshold": 85,
        },
        instructions=[
            "Runs VGGT with threshold 85 and saves `p_env_close.glb` automatically.",
        ],
        expected_globs=[
            "{SCOL}/working/Common/vggt_reconstruction/p_env_close.glb",
        ],
        timeout_sec=7200,
    ),
    StepSpec(
        key="glb_to_ply",
        label="Common › GLB → PLY conversion",
        script=MODELS_DIR / "glb_to_ply" / "glb_to_ply.py",
        env="vv_glb",
        config_template={
            "input_dir": "{SCOL}/working/Common/vggt_reconstruction",
            "output_dir": "{SCOL}/working/Common/vggt_reconstruction_ply",
            "files": ["p_env_far.glb", "p_env_close.glb"],
            "binary": True,
        },
        instructions=[
            "Converts VGGT outputs to PLY in `{SCOL}/working/Common/vggt_reconstruction_ply/`.",
        ],
        expected_globs=[
            "{SCOL}/working/Common/vggt_reconstruction_ply/*.ply",
        ],
        timeout_sec=600,
    ),
    StepSpec(
        key="far_denoise",
        label="Far › Denoising",
        script=MODELS_DIR / "denoising" / "denoising.py",
        env="vv_denoise",
        config_template={
            "input": "{SCOL}/working/Common/vggt_reconstruction_ply/p_env_far.ply",
            "output": "{SCOL}/working/far/denoising/p_env_far_denoised.ply",
        },
        instructions=[
            "Applies default SOR + ROR filters to clean the far point cloud.",
        ],
        expected_globs=[
            "{SCOL}/working/far/denoising/p_env_far_denoised.ply",
        ],
        timeout_sec=900,
    ),
    StepSpec(
        key="far_mesh",
        label="Far › Mesh reconstruction",
        script=MODELS_DIR / "meshlab" / "mesh_algo.py",
        env="vv_meshlab",
        config_template={
            "input": "{SCOL}/working/far/denoising/p_env_far_denoised.ply",
            "output": "{SCOL}/working/far/mesh_reconstruction/p_env_far_mesh.ply",
        },
        instructions=[
            "Generates the far mesh using PyMeshLab defaults.",
        ],
        expected_globs=[
            "{SCOL}/working/far/mesh_reconstruction/*.ply",
        ],
        timeout_sec=1800,
    ),
    StepSpec(
        key="near_mesh",
        label="Near › Mesh reconstruction",
        script=MODELS_DIR / "meshlab" / "mesh_algo.py",
        env="vv_meshlab",
        config_template={
            "input": "{SCOL}/working/Common/vggt_reconstruction_ply/p_env_close.ply",
            "output": "{SCOL}/working/near/mesh_reconstruction/p_env_close_mesh.ply",
        },
        instructions=[
            "Generates the near mesh using the same headless pipeline.",
        ],
        expected_globs=[
            "{SCOL}/working/near/mesh_reconstruction/*.ply",
        ],
        timeout_sec=1800,
    ),
    StepSpec(
        key="gnd_estimate",
        label="Far › Ground plane estimation (manual)",
        script=MODELS_DIR / "gnd_estimate" / "ground_plane_estimator_v2.py",
        env="vv_gnd_estimate",
        config_template={
            "input": "{SCOL}/working/far/mesh_reconstruction/p_env_far_mesh.ply",
            "output": "{SCOL}/final_output/gnd_estimated.json",
            "working_output": "{SCOL}/working/far/gnd_estimate/p_env_far_mesh_ground_plane.json",
            "working_dir": "{SCOL}/working/far/gnd_estimate",
        },

        # config_template={
        #     "input": "{SCOL}/working/far/mesh_reconstruction/p_env_far_mesh.ply",
        #     "output": "{SCOL}/final_output/gnd_estimated.json",
        #     "working_dir": "{SCOL}/working/far/gnd_estimate",
        # },
        instructions=[
            "Existing GUI step (manual interaction required).",
        ],
        expected_globs=[
            "{SCOL}/final_output/gnd_estimated.json",
        ],
        timeout_sec=1800,
    ),
    StepSpec(
        key="alignment",
        label="Near › Alignment (manual)",
        script=MODELS_DIR / "alignment" / "manual_alignment_v2.py",
        env="vv_mesh_alignment",
        config_template={
            "source": "{SCOL}/working/near/mesh_reconstruction/p_env_close_mesh.ply",
            "target": "{SCOL}/working/far/mesh_reconstruction/p_env_far_mesh.ply",
            "mesh": "{SCOL}/working/near/mesh_reconstruction/p_env_close_mesh.ply",
            "ground_json": "{SCOL}/final_output/gnd_estimated.json",
            "output": "{SCOL}/final_output/p_env_aligned.ply",
            "secondary_output": "{SCOL}/working/near/alignment/p_env_aligned.ply",
        },
        instructions=[
            "Existing GUI alignment (manual interaction required).",
        ],
        expected_globs=[
            "{SCOL}/final_output/*aligned*.ply",
            "{SCOL}/working/near/alignment/*aligned*.ply",
        ],
        timeout_sec=2400,
    ),
    StepSpec(
        key="final_visualization",
        label="Final › Visualization",
        script=MODELS_DIR / "final_visualization" / "ground_plane_estimator_simple.py",
        env="vv_mesh_alignment",
        config_template={
            "mesh": "{SCOL}/working/near/alignment/p_env_aligned.ply",
            "ground_json": "{SCOL}/working/far/gnd_estimate/p_env_far_mesh_ground_plane.json",
            "header": "Final Visualization for Doctor's Review",
        },
        instructions=[
            "Launches the final visualization window for clinical review.",
            "Mesh: `{SCOL}/working/near/alignment/p_env_aligned.ply`.",
            "Ground plane parameters: `{SCOL}/working/far/gnd_estimate/p_env_far_mesh_ground_plane.json`.",
        ],
        expected_globs=[
            "{SCOL}/working/near/alignment/p_env_aligned.ply",
        ],
        timeout_sec=1800,
    ),
]

STEP_BY_KEY = {s.key: s for s in PIPELINE_STEPS}
STEP_LABELS = [s.label for s in PIPELINE_STEPS]


def list_subdirs(path_str: str) -> List[str]:
    try:
        p = Path(path_str).expanduser().resolve()
        if not p.exists() or not p.is_dir():
            return []
        return sorted(str(x) for x in p.iterdir() if x.is_dir())
    except Exception:
        return []


def parent_path(path_str: str) -> str:
    try:
        p = Path(path_str).expanduser().resolve()
        return str(p.parent if p.parent != p else p)
    except Exception:
        return path_str


def ensure_scoliosis_structure(output_root: Path) -> Path:
    scol = output_root / "Scoliosis"
    structure = [
        scol / "input",
        scol / "working" / "input",
        scol / "working" / "Common" / "frames",
        scol / "working" / "Common" / "vggt_reconstruction",
        scol / "working" / "Common" / "vggt_reconstruction_ply",
        scol / "working" / "far" / "denoising",
        scol / "working" / "far" / "mesh_reconstruction",
        scol / "working" / "far" / "gnd_estimate",
        scol / "working" / "near" / "mesh_reconstruction",
        scol / "working" / "near" / "alignment",
        scol / "working" / "control",
        scol / "working" / "logs",
        scol / "final_output",
    ]
    for folder in structure:
        folder.mkdir(parents=True, exist_ok=True)
    return scol


def copy_required_inputs(input_folder: Path, scol_root: Path) -> List[str]:
    copied = []
    for name in ("p_env_far.mp4", "p_env_close.mp4"):
        src = input_folder / name
        if not src.exists():
            continue
        dests = [
            scol_root / "working" / "input" / name,
            scol_root / "input" / name,
        ]
        copied_this = False
        for dst in dests:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                shutil.copy2(src, dst)
                copied_this = True
        if copied_this:
            copied.append(name)
    return copied


def find_conda() -> Optional[str]:
    for exe in ("conda", "mamba"):
        if shutil.which(exe):
            return exe
    return None


def build_command(env_name: Optional[str], script: Path) -> List[str]:
    if env_name:
        conda = find_conda()
        if conda:
            return [conda, "run", "-n", env_name, "python", str(script)]
    return [sys.executable, str(script)]


def resolve_template(obj: Any, mapping: Dict[str, str]) -> Any:
    if isinstance(obj, str):
        value = obj
        for token, actual in mapping.items():
            value = value.replace(token, actual)
        return value
    if isinstance(obj, list):
        return [resolve_template(item, mapping) for item in obj]
    if isinstance(obj, dict):
        return {k: resolve_template(v, mapping) for k, v in obj.items()}
    return obj


def wait_for_artifacts(
    patterns: List[str],
    timeout: int,
    stop_event: threading.Event,
    earliest_mtime: Optional[float] = None,
    poll: float = 2.0,
) -> List[str]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if stop_event.is_set():
            return []
        matches: List[str] = []
        for pattern in patterns:
            for candidate in glob.glob(pattern):
                if earliest_mtime is not None:
                    try:
                        if Path(candidate).stat().st_mtime < earliest_mtime:
                            continue
                    except FileNotFoundError:
                        continue
                matches.append(candidate)
        if matches:
            return sorted(set(matches))
        time.sleep(poll)
    return []


class UIHandle:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.log_root = ROOT / "Scoliosis" / "working" / "logs"
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.step_logs: Dict[str, List[str]] = {s.key: [] for s in PIPELINE_STEPS}
        self.general_log: List[str] = []
        self.status_text: Dict[str, str] = {s.key: "IDLE" for s in PIPELINE_STEPS}
        self.processes: Dict[str, subprocess.Popen] = {}
        self.pipeline_running = False

    # -------- Log / status helpers --------
    def set_log_root(self, log_root: Path) -> None:
        with self.lock:
            self.log_root = log_root
            self.log_root.mkdir(parents=True, exist_ok=True)

    def reset_for_run(self, log_root: Path) -> None:
        with self.lock:
            self.log_root = log_root
            self.log_root.mkdir(parents=True, exist_ok=True)
            for file in self.log_root.glob("*"):
                if file.is_file():
                    try:
                        file.unlink()
                    except Exception:
                        pass
            self.step_logs = {s.key: [] for s in PIPELINE_STEPS}
            self.general_log = []
            self.status_text = {s.key: "IDLE" for s in PIPELINE_STEPS}
            self._write_status_file()

    def append_general(self, message: str) -> None:
        with self.lock:
            self.general_log.append(message.rstrip() + "\n")
            general_path = self.log_root / "general.log"
            general_path.write_text("".join(self.general_log)[-200000:])

    def read_general(self) -> str:
        try:
            return (self.log_root / "general.log").read_text()
        except Exception:
            return "".join(self.general_log)[-200000:]

    def append_step_log(self, key: str, text: str) -> None:
        with self.lock:
            self.step_logs.setdefault(key, []).append(text)
            log_path = self.log_root / f"{key}.log"
            log_path.write_text("".join(self.step_logs[key])[-200000:])

    def read_step_log(self, key: str) -> str:
        try:
            return (self.log_root / f"{key}.log").read_text()[-200000:]
        except Exception:
            return "".join(self.step_logs.get(key, []))[-200000:]

    def update_status(self, key: str, status: str, detail: str = "") -> None:
        suffix = f" – {detail}" if detail else ""
        text = f"{status}{suffix}"
        with self.lock:
            self.status_text[key] = text
            self._write_status_file()

    def get_status_list(self) -> List[str]:
        with self.lock:
            return [self.status_text.get(step.key, "IDLE") for step in PIPELINE_STEPS]

    def _write_status_file(self) -> None:
        status_path = self.log_root / "status.json"
        status_path.write_text(json.dumps(self.status_text, indent=2))

    def set_pipeline_running(self, value: bool) -> None:
        with self.lock:
            self.pipeline_running = value

    def is_pipeline_running(self) -> bool:
        with self.lock:
            return self.pipeline_running

    # -------- process management --------
    def attach_process(self, key: str, proc: subprocess.Popen) -> None:
        with self.lock:
            self.processes[key] = proc

    def detach_process(self, key: str) -> None:
        with self.lock:
            self.processes.pop(key, None)

    def terminate_process(self, key: str) -> None:
        with self.lock:
            proc = self.processes.get(key)
        if not proc:
            return
        terminate_process(proc)

    def terminate_all(self) -> None:
        with self.lock:
            procs = list(self.processes.values())
            self.processes.clear()
        for proc in procs:
            terminate_process(proc)


def terminate_process(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


class PipelineRunner(threading.Thread):
    def __init__(self, input_folder: Path, output_root: Path, mode: str, start_index: int, ui_handle: UIHandle) -> None:
        super().__init__(daemon=True)
        self.input_folder = Path(input_folder).expanduser().resolve()
        self.output_root = Path(output_root).expanduser().resolve()
        self.mode = mode
        self.start_index = start_index
        self.ui = ui_handle
        self.stop_event = threading.Event()
        self.scol_root: Optional[Path] = None
        self.run_start_ts: float = time.time()

    def stop(self) -> None:
        self.stop_event.set()
        self.ui.append_general("Stop request received – terminating active processes.")
        self.ui.terminate_all()

    def _replace_mapping(self) -> Dict[str, str]:
        assert self.scol_root is not None
        return {
            "{INPUT}": str(self.input_folder),
            "{OUTPUT_ROOT}": str(self.output_root),
            "{SCOL}": str(self.scol_root),
        }

    def run(self) -> None:
        self.ui.set_pipeline_running(True)
        try:
            if not self.prepare_environment():
                return
            for idx in range(self.start_index, len(PIPELINE_STEPS)):
                if self.stop_event.is_set():
                    self.ui.append_general("Pipeline aborted by user.")
                    return
                step = PIPELINE_STEPS[idx]
                success = self.execute_step(step)
                if not success:
                    return
                if self.mode == "semi-automatic" and idx + 1 < len(PIPELINE_STEPS):
                    self.ui.append_general(
                        "Semi-automatic mode: select the next step in the dropdown and press 'Run Pipeline' to continue."
                    )
                    self.ui.set_pipeline_running(False)
                    return
            self.ui.append_general("Pipeline completed successfully.")
        finally:
            self.ui.set_pipeline_running(False)

    def prepare_environment(self) -> bool:
        missing = [name for name in ("p_env_far.mp4", "p_env_close.mp4") if not (self.input_folder / name).exists()]
        if missing:
            self.ui.append_general(f"Missing required video files in input folder: {missing}")
            return False
        self.scol_root = ensure_scoliosis_structure(self.output_root)
        log_root = self.scol_root / "working" / "logs"
        self.ui.reset_for_run(log_root)
        copied = copy_required_inputs(self.input_folder, self.scol_root)
        if copied:
            self.ui.append_general(f"Copied fresh inputs into Scoliosis/working/input: {copied}")
        else:
            self.ui.append_general("Input videos already present in Scoliosis/working/input.")
        self.run_start_ts = time.time()
        for idx, step in enumerate(PIPELINE_STEPS):
            if idx < self.start_index:
                self.ui.update_status(step.key, "SKIPPED", "Skipped (start index)")
            else:
                self.ui.update_status(step.key, "IDLE")
        self.ui.append_general(
            f"Environment ready. Output workspace: {self.scol_root}. Starting from step index {self.start_index}."
        )
        return True

    def execute_step(self, step: StepSpec) -> bool:
        assert self.scol_root is not None
        mapping = self._replace_mapping()
        resolved_config = resolve_template(step.config_template, mapping)
        resolved_instructions = resolve_template(step.instructions, mapping)
        cfg = {
            "step": step.key,
            "label": step.label,
            "script": str(step.script),
            "environment": step.env,
            "instructions": resolved_instructions,
            "expected_globs": resolve_template(step.expected_globs, mapping),
        }
        if isinstance(resolved_config, dict):
            cfg.update(resolved_config)
        cfg["parameters"] = resolved_config
        control_dir = self.scol_root / "working" / "control"
        control_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = control_dir / f"{step.key}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))

        instruction_lines = "\n".join(f"- {line}" for line in resolved_instructions)
        self.ui.append_general(
            f"{step.label}\nConfig: {cfg_path}\nFollow these instructions:\n{instruction_lines}"
        )
        self.ui.update_status(step.key, "RUNNING", "Tool launched")

        if not step.script.exists():
            self.ui.append_general(f"Script not found: {step.script}")
            self.ui.update_status(step.key, "FAILED", "Script missing")
            return False

        try:
            cmd = build_command(step.env, step.script)
            cmd += ["--config", str(cfg_path)]
            self.ui.append_step_log(step.key, f"[launcher] Command: {' '.join(cmd)}\n")
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as exc:
            self.ui.append_general(f"Failed to launch {step.label}: {exc}")
            self.ui.update_status(step.key, "FAILED", "Launch error")
            return False

        self.ui.attach_process(step.key, proc)

        reader = threading.Thread(
            target=self._stream_stdout, args=(step.key, proc), name=f"{step.key}_stdout", daemon=True
        )
        reader.start()

        patterns = resolve_template(step.expected_globs, mapping)
        artifacts = wait_for_artifacts(
            patterns, step.timeout_sec, self.stop_event, earliest_mtime=self.run_start_ts
        )

        if self.stop_event.is_set():
            self.ui.update_status(step.key, "ABORTED", "Stopped by user")
            self.ui.detach_process(step.key)
            terminate_process(proc)
            return False

        if artifacts:
            artifact_preview = ", ".join(Path(a).name for a in artifacts[:3])
            self.ui.update_status(step.key, "DONE", artifact_preview)
            self.ui.append_general(f"{step.label}: detected artifacts {artifact_preview}")
            self.ui.detach_process(step.key)
            terminate_process(proc)
            return True

        # No artifacts detected – check process status
        exit_code = proc.poll()
        if exit_code is None:
            self.ui.append_general(
                f"{step.label} timed out after {step.timeout_sec} seconds without matching artifacts {patterns}."
            )
            self.ui.update_status(step.key, "FAILED", "Timeout waiting for outputs")
            terminate_process(proc)
        else:
            self.ui.append_general(
                f"{step.label} ended with exit code {exit_code} before expected artifacts appeared."
            )
            self.ui.update_status(step.key, "FAILED", f"Process exited ({exit_code})")
        self.ui.detach_process(step.key)
        return False

    def _stream_stdout(self, key: str, proc: subprocess.Popen) -> None:
        if not proc.stdout:
            return
        try:
            for line in proc.stdout:
                self.ui.append_step_log(key, line)
        except Exception as exc:
            self.ui.append_step_log(key, f"[launcher] stdout reader error: {exc}\n")


def build_app() -> gr.Blocks:
    ui = UIHandle()

    STATUS_PANEL_CSS = """
    .status-panel {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 12px 14px;
    }
    .status-item {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
    }
    .status-item:last-child {
        margin-bottom: 0;
    }
    .status-check {
        width: 18px;
        text-align: center;
        font-size: 1.1rem;
        color: #2563eb;
    }
    .status-item.done .status-check {
        color: #16a34a;
    }
    .status-item.failed .status-check {
        color: #dc2626;
    }
    .status-dot {
        width: 12px;
        height: 12px;
        border-radius: 999px;
        background: #d1d5db;
        display: inline-block;
    }
    .status-item.running .status-dot {
        background: #16a34a;
        box-shadow: 0 0 0 4px rgba(22, 163, 74, 0.3);
        animation: status-pulse 1.5s ease-out infinite;
    }
    .status-item.failed .status-dot {
        background: #dc2626;
    }
    .status-item.done .status-dot {
        background: #16a34a;
    }
    .status-item.pending .status-dot {
        background: #f59e0b;
    }
    .status-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #111827;
    }
    @keyframes status-pulse {
        0% { box-shadow: 0 0 0 0 rgba(22, 163, 74, 0.35); }
        70% { box-shadow: 0 0 0 8px rgba(22, 163, 74, 0); }
        100% { box-shadow: 0 0 0 0 rgba(22, 163, 74, 0); }
    }
    """

    with gr.Blocks(title="Scoliosis Pipeline Launcher", css=STATUS_PANEL_CSS) as demo:
        gr.Markdown("## Scoliosis Pipeline — Unified Launcher")

        with gr.Row():
            input_path = gr.Textbox(
                label="Input folder (contains p_env_far.mp4 & p_env_close.mp4)",
                placeholder="/path/to/input",
                lines=1,
            )
            output_root = gr.Textbox(
                label="Output root (a Scoliosis/ folder is created inside)",
                placeholder="/path/to/output_root",
                lines=1,
            )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input folder navigator")
                in_nav_path = gr.Textbox(value=str(Path.home()), label="Current input path", lines=1)
                in_nav_dropdown = gr.Dropdown(choices=list_subdirs(str(Path.home())), label="Subfolders", interactive=True)
                with gr.Row():
                    in_nav_up = gr.Button("Up")
                    in_nav_refresh = gr.Button("Refresh")
                    in_nav_choose = gr.Button("Choose this folder")
            with gr.Column():
                gr.Markdown("### Output root navigator")
                out_nav_path = gr.Textbox(value=str(Path.home()), label="Current output path", lines=1)
                out_nav_dropdown = gr.Dropdown(
                    choices=list_subdirs(str(Path.home())), label="Subfolders", interactive=True
                )
                with gr.Row():
                    out_nav_up = gr.Button("Up")
                    out_nav_refresh = gr.Button("Refresh")
                    out_nav_choose = gr.Button("Choose this folder")

        with gr.Row():
            start_step = gr.Dropdown(choices=STEP_LABELS, value=STEP_LABELS[0], label="Start from step")
            run_btn = gr.Button("Run Pipeline")
            stop_btn = gr.Button("Stop Pipeline")

        def render_status_panel_local() -> str:
            statuses = ui.get_status_list()
            rows = ['<div class="status-panel">']
            for idx, (spec, status_text) in enumerate(zip(PIPELINE_STEPS, statuses), start=1):
                status_text = status_text or "IDLE"
                upper = status_text.upper()
                css_class = "idle"
                checkbox = "&#x2610;"  # ☐
                if any(token in upper for token in ("FAILED", "ERROR", "ABORTED")):
                    css_class = "failed"
                    checkbox = "&#x2612;"  # ☒
                elif any(token in upper for token in ("DONE", "COMPLETED", "SKIPPED")):
                    css_class = "done"
                    checkbox = "&#x2611;"  # ☑
                elif any(token in upper for token in ("RUNNING", "LAUNCHED")):
                    css_class = "running"
                elif "PENDING" in upper:
                    css_class = "pending"

                label_html = html.escape(spec.label)
                rows.append(
                    f'<div class="status-item {css_class}">'
                    f'<span class="status-check">{checkbox}</span>'
                    f'<span class="status-dot"></span>'
                    f'<span class="status-title">{idx}. {label_html}</span>'
                    f"</div>"
                )
            rows.append("</div>")
            return "\n".join(rows)

        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### Pipeline tracker")
                status_sidebar = gr.HTML(render_status_panel_local())
                refresh_status_btn = gr.Button("Refresh tracker")
            with gr.Column(scale=3):
                with gr.Row():
                    log_step = gr.Dropdown(choices=STEP_LABELS, value=STEP_LABELS[0], label="Show step log")
                    refresh_step_log_btn = gr.Button("Refresh step log")
                step_log = gr.Textbox(label="Selected step log", lines=14)

                general_log = gr.Textbox(label="General log", lines=8)
                refresh_general_btn = gr.Button("Refresh general log")

        runner_holder: Dict[str, Optional[PipelineRunner]] = {"runner": None}

        def on_run(input_folder: str, output_folder: str, start_label: str) -> str:
            if ui.is_pipeline_running():
                return "Pipeline already running – stop it before starting again."
            if not input_folder or not output_folder:
                return "Select both input and output folders before running."
            try:
                start_index = STEP_LABELS.index(start_label)
            except ValueError:
                start_index = 0
            runner = PipelineRunner(Path(input_folder), Path(output_folder), "fully-automatic", start_index, ui)
            runner_holder["runner"] = runner
            runner.start()
            time.sleep(0.1)
            return ui.read_general()

        def on_stop() -> str:
            runner = runner_holder.get("runner")
            if runner and runner.is_alive():
                runner.stop()
                runner_holder["runner"] = None
                return "Stop requested."
            return "No active pipeline to stop."

        def on_refresh_general() -> str:
            return ui.read_general()

        def on_refresh_step_log(label: str) -> str:
            try:
                idx = STEP_LABELS.index(label)
            except ValueError:
                idx = 0
            key = PIPELINE_STEPS[idx].key
            return ui.read_step_log(key)

        def on_select_folder(choice: str):
            if not choice:
                return gr.update(value=""), gr.update(choices=[])
            new_path = Path(choice).expanduser().resolve()
            return gr.update(value=str(new_path)), gr.update(choices=list_subdirs(str(new_path)))

        def on_up(path_str: str):
            parent = parent_path(path_str)
            return gr.update(value=parent), gr.update(choices=list_subdirs(parent))

        def on_refresh(path_str: str):
            return gr.update(choices=list_subdirs(path_str))

        def on_choose(path_str: str):
            return path_str

        run_btn.click(on_run, inputs=[input_path, output_root, start_step], outputs=[general_log])
        stop_btn.click(on_stop, inputs=[], outputs=[general_log])

        refresh_status_btn.click(render_status_panel_local, inputs=[], outputs=[status_sidebar])

        refresh_general_btn.click(on_refresh_general, inputs=[], outputs=[general_log])
        refresh_step_log_btn.click(on_refresh_step_log, inputs=[log_step], outputs=[step_log])

        in_nav_dropdown.change(on_select_folder, inputs=[in_nav_dropdown], outputs=[in_nav_path, in_nav_dropdown])
        out_nav_dropdown.change(on_select_folder, inputs=[out_nav_dropdown], outputs=[out_nav_path, out_nav_dropdown])

        in_nav_up.click(on_up, inputs=[in_nav_path], outputs=[in_nav_path, in_nav_dropdown])
        out_nav_up.click(on_up, inputs=[out_nav_path], outputs=[out_nav_path, out_nav_dropdown])

        in_nav_refresh.click(on_refresh, inputs=[in_nav_path], outputs=[in_nav_dropdown])
        out_nav_refresh.click(on_refresh, inputs=[out_nav_path], outputs=[out_nav_dropdown])

        in_nav_choose.click(on_choose, inputs=[in_nav_path], outputs=[input_path])
        out_nav_choose.click(on_choose, inputs=[out_nav_path], outputs=[output_root])

        interval = None
        try:
            interval = gr.Interval(3000)
        except AttributeError:
            interval = None
        if interval is not None:
            interval.change(render_status_panel_local, inputs=[], outputs=[status_sidebar])
            interval.change(on_refresh_general, inputs=[], outputs=[general_log])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7862, share=False)
