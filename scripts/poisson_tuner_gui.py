#!/usr/bin/env python3
import os
import subprocess
import threading
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "poisson_tuner"
DEFAULT_MESH_NAME = "mesh_poisson_after_stage3.ply"
DEFAULT_MESH_CANDIDATES = [
    "mesh_poisson_after_stage3.ply",
    "mesh_poisson_after_stage2.ply",
    "mesh_poisson_after_stage1.ply",
    "mesh_poisson_after_iter1.ply",
    "mesh_poisson_init.ply",
]


def build_env(values):
    env = os.environ.copy()
    env.update({
        "SONAR_OUTPUT_DIR": str(values["output_dir"]),
        "SONAR_STAGE2_ITERS": str(values["stage2_iters"]),
        "POISSON_DEPTH": str(values["poisson_depth"]),
        "POISSON_DENSITY_QUANTILE": f"{values['density_quantile']:.4f}",
        "POISSON_MIN_OPACITY": f"{values['min_opacity']:.4f}",
        "POISSON_OPACITY_PERCENTILE": f"{values['opacity_percentile']:.4f}",
        "POISSON_SCALE_PERCENTILE": f"{values['scale_percentile']:.4f}",
    })
    return env


def run_debug_multiframe(values, log_callback=None):
    cmd = ["python", "debug_multiframe.py"]
    env = build_env(values)
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if log_callback is not None and proc.stdout is not None:
        for line in proc.stdout:
            log_callback(line.rstrip())
    proc.wait()
    return proc.returncode


def load_mesh(path):
    if not path.exists():
        return None
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        return None
    mesh.compute_vertex_normals()
    return mesh


class PoissonTunerApp:
    def __init__(self):
        self.app = gui.Application.instance
        self.app.initialize()

        self.window = gui.Application.instance.create_window("Poisson Tuner", 1280, 860)
        self.window.set_on_close(self._on_close)

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.05, 0.05, 0.05, 1.0])

        self.panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))
        self.panel.add_child(gui.Label("Poisson Tuner"))

        self.output_dir = gui.TextEdit()
        self.output_dir.text_value = str(DEFAULT_OUTPUT_DIR)
        self.output_dir.enabled = False
        self.panel.add_child(gui.Label("Output Directory"))
        self.panel.add_child(self.output_dir)

        self.stage2_iters = gui.Slider(gui.Slider.INT)
        self.stage2_iters.set_limits(10, 5000)
        self.stage2_iters.int_value = 1000
        self.panel.add_child(gui.Label("Stage 2 Iterations"))
        self.panel.add_child(self.stage2_iters)

        self.poisson_depth = gui.Slider(gui.Slider.INT)
        self.poisson_depth.set_limits(6, 12)
        self.poisson_depth.int_value = 9
        self.panel.add_child(gui.Label("Poisson Depth"))
        self.panel.add_child(self.poisson_depth)

        self.density_quantile = gui.Slider(gui.Slider.DOUBLE)
        self.density_quantile.set_limits(0.0, 0.2)
        self.density_quantile.double_value = 0.02
        self.panel.add_child(gui.Label("Density Quantile"))
        self.panel.add_child(self.density_quantile)

        self.min_opacity = gui.Slider(gui.Slider.DOUBLE)
        self.min_opacity.set_limits(0.0, 0.5)
        self.min_opacity.double_value = 0.05
        self.panel.add_child(gui.Label("Min Opacity"))
        self.panel.add_child(self.min_opacity)

        self.opacity_percentile = gui.Slider(gui.Slider.DOUBLE)
        self.opacity_percentile.set_limits(0.0, 0.9)
        self.opacity_percentile.double_value = 0.2
        self.panel.add_child(gui.Label("Opacity Percentile"))
        self.panel.add_child(self.opacity_percentile)

        self.scale_percentile = gui.Slider(gui.Slider.DOUBLE)
        self.scale_percentile.set_limits(0.1, 1.0)
        self.scale_percentile.double_value = 0.9
        self.panel.add_child(gui.Label("Scale Percentile"))
        self.panel.add_child(self.scale_percentile)

        self.mesh_name = gui.TextEdit()
        self.mesh_name.text_value = DEFAULT_MESH_NAME
        self.mesh_name.enabled = False
        self.panel.add_child(gui.Label("Mesh File"))
        self.panel.add_child(self.mesh_name)

        self.mesh_note = gui.Label("(auto-loads first available Poisson mesh)")
        self.panel.add_child(self.mesh_note)

        self.run_button = gui.Button("Run")
        self.run_button.set_on_clicked(self._on_run)
        self.panel.add_child(self.run_button)

        self.reload_button = gui.Button("Reload Mesh")
        self.reload_button.set_on_clicked(self._on_reload)
        self.panel.add_child(self.reload_button)

        self.log_box = gui.TextEdit()
        self.log_box.text_value = "Ready. Click Run to generate mesh."
        self.panel.add_child(gui.Label("Run Log"))
        self.panel.add_child(self.log_box)

        self.mesh_info = gui.Label("Mesh: (not loaded)")
        self.panel.add_child(self.mesh_info)

        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)

        self._mesh_geometry = None
        self._is_running = False

    def _on_layout(self, layout_context):
        content_rect = self.window.content_rect
        panel_width = 360
        self.panel.frame = gui.Rect(content_rect.x, content_rect.y, panel_width, content_rect.height)
        self.scene_widget.frame = gui.Rect(
            content_rect.x + panel_width,
            content_rect.y,
            content_rect.width - panel_width,
            content_rect.height,
        )

    def _collect_values(self):
        return {
            "output_dir": Path(self.output_dir.text_value.strip() or str(DEFAULT_OUTPUT_DIR)),
            "stage2_iters": int(self.stage2_iters.int_value),
            "poisson_depth": int(self.poisson_depth.int_value),
            "density_quantile": float(self.density_quantile.double_value),
            "min_opacity": float(self.min_opacity.double_value),
            "opacity_percentile": float(self.opacity_percentile.double_value),
            "scale_percentile": float(self.scale_percentile.double_value),
            "mesh_name": self.mesh_name.text_value.strip() or DEFAULT_MESH_NAME,
        }

    def _append_log(self, text):
        def update():
            self.log_box.text_value += f"\n{text}"
        gui.Application.instance.post_to_main_thread(self.window, update)

    def _on_run(self):
        if self._is_running:
            self._append_log("Run already in progress...")
            return

        values = self._collect_values()
        output_dir = Path(values["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        self.log_box.text_value = (
            f"Running debug_multiframe.py (output: {output_dir})...\n"
            f"Looking for: {DEFAULT_MESH_NAME}"
        )
        self._is_running = True

        def task():
            code = run_debug_multiframe(values, log_callback=self._append_log)
            self._append_log(f"Run finished with code {code}")
            self._is_running = False
            self._load_mesh(values)

        threading.Thread(target=task, daemon=True).start()

    def _on_reload(self):
        values = self._collect_values()
        self._append_log("Reloading mesh...")
        self._load_mesh(values)

    def _load_mesh(self, values):
        mesh_path = Path(values["output_dir"]) / values["mesh_name"]
        candidate_paths = [mesh_path]
        if values["mesh_name"] == DEFAULT_MESH_NAME:
            candidate_paths = [Path(values["output_dir"]) / name for name in DEFAULT_MESH_CANDIDATES]

        found_path = None
        mesh = None
        for path in candidate_paths:
            mesh = load_mesh(path)
            if mesh is not None:
                found_path = path
                break

        if mesh is None:
            self._append_log(f"Mesh not found or empty. Looked in: {values['output_dir']}")
            def update_label():
                self.mesh_info.text = "Mesh: (not loaded)"
            gui.Application.instance.post_to_main_thread(self.window, update_label)
            return

        def update_scene():
            self.scene_widget.scene.clear_geometry()
            material = rendering.MaterialRecord()
            material.shader = "defaultLit"
            self.scene_widget.scene.add_geometry("mesh", mesh, material)
            bounds = mesh.get_axis_aligned_bounding_box()
            self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())
            self.scene_widget.scene.show_axes(True)
            if found_path is not None:
                self.mesh_info.text = f"Mesh: {found_path.name}"
                self.log_box.text_value += f"\nLoaded mesh: {found_path}"

        gui.Application.instance.post_to_main_thread(self.window, update_scene)

    def _on_close(self):
        return True

    def run(self):
        self.app.run()


if __name__ == "__main__":
    app = PoissonTunerApp()
    app.run()
