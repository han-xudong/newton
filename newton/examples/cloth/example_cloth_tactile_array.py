# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Tactile Array
#
# Demonstrates a clamped tactile cloth patch instrumented with a taxel array.
# A rigid sphere is dropped onto the patch, the cloth reads the impact through
# SensorTactileArray, and the viewer shows live force curves together with a
# taxel-force overlay on the cloth surface.
#
# Command: python -m newton.examples cloth_tactile_array --cloth-verts-width 11 --cloth-verts-length 9 --cloth-width 0.25 --cloth-length 0.20
#
###########################################################################

from __future__ import annotations

import math
from pathlib import Path
import warnings

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.sensors import SensorTactileArray
from newton.tests.unittest_utils import find_nonfinite_members


class Example:
    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--sim-substeps",
            type=int,
            default=8,
            help="Number of simulation substeps per rendered frame.",
        )
        parser.add_argument(
            "--solver-iterations",
            type=int,
            default=10,
            help="Number of VBD solver iterations per substep.",
        )
        parser.add_argument(
            "--cloth-verts-width",
            type=int,
            default=21,
            help="Cloth sensing vertex resolution across cloth width [count].",
        )
        parser.add_argument(
            "--cloth-verts-length",
            type=int,
            default=21,
            help="Cloth sensing vertex resolution across cloth length [count].",
        )
        parser.add_argument(
            "--cloth-width",
            type=float,
            default=0.20,
            help="Physical cloth size across width [m].",
        )
        parser.add_argument(
            "--cloth-length",
            type=float,
            default=0.20,
            help="Physical cloth size across length [m].",
        )
        parser.add_argument(
            "--cloth-height",
            type=float,
            default=0.3,
            help="Cloth plane height above the ground [m].",
        )
        parser.add_argument(
            "--cloth-particle-mass",
            type=float,
            default=0.002,
            help="Mass per cloth particle [kg].",
        )
        parser.add_argument(
            "--cloth-tri-ke",
            type=float,
            default=1.0e3,
            help="Cloth triangle stretch stiffness.",
        )
        parser.add_argument(
            "--cloth-tri-ka",
            type=float,
            default=1.0e3,
            help="Cloth triangle area/shear stiffness.",
        )
        parser.add_argument(
            "--cloth-tri-kd",
            type=float,
            default=8.0e-4,
            help="Cloth triangle damping.",
        )
        parser.add_argument(
            "--cloth-edge-ke",
            type=float,
            default=3.0e-2,
            help="Cloth edge bending stiffness.",
        )
        parser.add_argument(
            "--cloth-edge-kd",
            type=float,
            default=2.0e-3,
            help="Cloth edge damping.",
        )
        parser.add_argument(
            "--cloth-particle-radius",
            type=float,
            default=0.012,
            help="Cloth particle radius [m].",
        )
        parser.add_argument(
            "--ball-drop-height",
            type=float,
            default=0.4,
            help="Initial ball center height above the ground [m].",
        )
        parser.add_argument(
            "--ball-radius",
            type=float,
            default=0.03,
            help="Ball radius [m].",
        )
        parser.add_argument(
            "--ball-mass",
            type=float,
            default=1.0,
            help="Ball mass [kg].",
        )
        parser.add_argument(
            "--ball-ke",
            type=float,
            default=2.0e5,
            help="Ball contact stiffness.",
        )
        parser.add_argument(
            "--ball-kd",
            type=float,
            default=1.0e-3,
            help="Ball contact damping.",
        )
        parser.add_argument(
            "--ball-mu",
            type=float,
            default=0.2,
            help="Ball friction coefficient.",
        )
        parser.add_argument(
            "--array-cols",
            type=int,
            default=21,
            help="Tactile array output resolution across cloth width [count].",
        )
        parser.add_argument(
            "--array-rows",
            type=int,
            default=21,
            help="Tactile array output resolution across cloth length [count].",
        )
        parser.add_argument(
            "--soft-contact-margin",
            type=float,
            default=0.012,
            help="Soft contact margin used for cloth-ball sensing [m].",
        )
        parser.add_argument(
            "--soft-contact-ke",
            type=float,
            default=2.0e4,
            help="Model soft-contact stiffness.",
        )
        parser.add_argument(
            "--soft-contact-kd",
            type=float,
            default=1.0e-5,
            help="Model soft-contact damping.",
        )
        parser.add_argument(
            "--soft-contact-mu",
            type=float,
            default=0.2,
            help="Model soft-contact friction coefficient.",
        )
        parser.add_argument(
            "--record-file",
            type=str,
            default=None,
            help="Optional replay recording path. When set, the example records to a ViewerFile and saves a .bin replay.",
        )
        return parser

    @staticmethod
    def create_frame_mesh(
        outer_half_x: float,
        outer_half_y: float,
        inner_half_x: float,
        inner_half_y: float,
        top_z: float,
        bottom_z: float,
    ) -> newton.Mesh:
        vertices = np.array(
            [
                [-outer_half_x, -outer_half_y, top_z],
                [outer_half_x, -outer_half_y, top_z],
                [outer_half_x, outer_half_y, top_z],
                [-outer_half_x, outer_half_y, top_z],
                [-inner_half_x, -inner_half_y, top_z],
                [inner_half_x, -inner_half_y, top_z],
                [inner_half_x, inner_half_y, top_z],
                [-inner_half_x, inner_half_y, top_z],
                [-outer_half_x, -outer_half_y, bottom_z],
                [outer_half_x, -outer_half_y, bottom_z],
                [outer_half_x, outer_half_y, bottom_z],
                [-outer_half_x, outer_half_y, bottom_z],
                [-inner_half_x, -inner_half_y, bottom_z],
                [inner_half_x, -inner_half_y, bottom_z],
                [inner_half_x, inner_half_y, bottom_z],
                [-inner_half_x, inner_half_y, bottom_z],
            ],
            dtype=np.float32,
        )
        indices = np.array(
            [
                0, 1, 5,
                0, 5, 4,
                1, 2, 6,
                1, 6, 5,
                2, 3, 7,
                2, 7, 6,
                3, 0, 4,
                3, 4, 7,
                8, 13, 9,
                8, 12, 13,
                9, 14, 10,
                9, 13, 14,
                10, 15, 11,
                10, 14, 15,
                11, 12, 8,
                11, 15, 12,
                0, 9, 1,
                0, 8, 9,
                1, 10, 2,
                1, 9, 10,
                2, 11, 3,
                2, 10, 11,
                3, 8, 0,
                3, 11, 8,
                4, 5, 13,
                4, 13, 12,
                5, 6, 14,
                5, 14, 13,
                6, 7, 15,
                6, 15, 14,
                7, 4, 12,
                7, 12, 15,
            ],
            dtype=np.int32,
        )
        return newton.Mesh(vertices, indices, compute_inertia=False, is_solid=False)

    @staticmethod
    def camera_angles_from_target(camera_pos: np.ndarray, target_pos: np.ndarray) -> tuple[float, float]:
        view_dir = np.asarray(target_pos - camera_pos, dtype=np.float64)
        planar_distance = float(np.hypot(view_dir[0], view_dir[1]))
        pitch = math.degrees(math.atan2(float(view_dir[2]), max(planar_distance, 1.0e-8)))
        yaw = math.degrees(math.atan2(float(view_dir[1]), float(view_dir[0])))
        return pitch, yaw

    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = int(getattr(args, "sim_substeps", 8))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.solver_iterations = int(getattr(args, "solver_iterations", 10))

        self.viewer = viewer
        self._viewer_is_null = isinstance(self.viewer, newton.viewer.ViewerNull)

        self.taxel_rows = int(getattr(args, "array_rows", 8))
        self.taxel_cols = int(getattr(args, "array_cols", 10))
        self.cloth_cols = int(getattr(args, "cloth_verts_width", 11))
        self.cloth_rows = int(getattr(args, "cloth_verts_length", 9))
        self.cloth_cells_x = self.cloth_cols - 1
        self.cloth_cells_y = self.cloth_rows - 1
        self.cloth_width = float(getattr(args, "cloth_width", 0.25))
        self.cloth_length = float(getattr(args, "cloth_length", 0.20))
        self.soft_contact_margin = float(getattr(args, "soft_contact_margin", 0.012))
        self.width = self.cloth_width
        self.length = self.cloth_length
        self.cell_x = self.width / float(self.cloth_cells_x)
        self.cell_y = self.length / float(self.cloth_cells_y)
        self.cloth_height = float(getattr(args, "cloth_height", 0.3))
        self.cloth_particle_mass = float(getattr(args, "cloth_particle_mass", 0.002))
        self.cloth_tri_ke = float(getattr(args, "cloth_tri_ke", 6.0e4))
        self.cloth_tri_ka = float(getattr(args, "cloth_tri_ka", 6.0e4))
        self.cloth_tri_kd = float(getattr(args, "cloth_tri_kd", 8.0e-4))
        self.cloth_edge_ke = float(getattr(args, "cloth_edge_ke", 3.0e-2))
        self.cloth_edge_kd = float(getattr(args, "cloth_edge_kd", 2.0e-3))
        self.cloth_particle_radius = float(getattr(args, "cloth_particle_radius", 0.012))
        self.ball_radius = float(getattr(args, "ball_radius", 0.02))
        self.ball_mass = float(getattr(args, "ball_mass", 0.2))
        self.ball_ke = float(getattr(args, "ball_ke", 2.0e5))
        self.ball_kd = float(getattr(args, "ball_kd", 1.0e-2))
        self.ball_mu = float(getattr(args, "ball_mu", 0.45))
        self.ball_drop_height = float(getattr(args, "ball_drop_height", 0.4))
        self.soft_contact_ke = float(getattr(args, "soft_contact_ke", 2.0e4))
        self.soft_contact_kd = float(getattr(args, "soft_contact_kd", 1.0e-4))
        self.soft_contact_mu = float(getattr(args, "soft_contact_mu", 0.6))
        if self.sim_substeps <= 0:
            raise ValueError("`sim_substeps` must be positive.")
        if self.solver_iterations <= 0:
            raise ValueError("`solver_iterations` must be positive.")
        if self.taxel_rows <= 0 or self.taxel_cols <= 0:
            raise ValueError("`array_rows` and `array_cols` must be positive.")
        if self.cloth_cols < 2 or self.cloth_rows < 2:
            raise ValueError("`cloth_verts_width` and `cloth_verts_length` must be at least 2.")
        if self.cloth_width <= 0.0 or self.cloth_length <= 0.0:
            raise ValueError("`cloth_width` and `cloth_length` must be positive.")
        if self.cell_x <= 0.0 or self.cell_y <= 0.0:
            raise ValueError("Derived cloth cell sizes must be positive.")
        if self.cloth_particle_mass <= 0.0:
            raise ValueError("`cloth_particle_mass` must be positive.")
        if self.cloth_particle_radius <= 0.0:
            raise ValueError("`cloth_particle_radius` must be positive.")
        if self.soft_contact_margin < 0.0:
            raise ValueError("`soft_contact_margin` must be non-negative.")
        if self.ball_ke < 0.0 or self.ball_kd < 0.0 or self.ball_mu < 0.0:
            raise ValueError("`ball_ke`, `ball_kd`, and `ball_mu` must be non-negative.")
        if self.soft_contact_ke < 0.0 or self.soft_contact_kd < 0.0 or self.soft_contact_mu < 0.0:
            raise ValueError("`soft_contact_ke`, `soft_contact_kd`, and `soft_contact_mu` must be non-negative.")
        if self.ball_radius <= 0.0:
            raise ValueError("`ball_radius` must be positive.")
        if self.ball_mass <= 0.0:
            raise ValueError("`ball_mass` must be positive.")
        minimum_drop_height = self.cloth_height + self.ball_radius + 0.01
        if self.ball_drop_height <= minimum_drop_height:
            raise ValueError(
                f"`ball_drop_height` must be greater than {minimum_drop_height:.3f} m so the ball starts above the cloth."
            )
        if self.taxel_rows > self.cloth_rows or self.taxel_cols > self.cloth_cols:
            warnings.warn(
                "Tactile array resolution exceeds the cloth sensing vertex grid on at least one axis; this only densifies the displayed taxel map and does not increase physical sensing fidelity.",
                stacklevel=2,
            )
        self.frame_height = min(
            self.ball_drop_height - 0.03,
            max(self.cloth_height + 0.20, self.ball_drop_height - max(0.08, 0.75 * self.ball_radius)),
        )
        self.cloth_origin = np.array(
            [-0.5 * self.width, -0.5 * self.length, self.cloth_height],
            dtype=np.float32,
        )
        self.ball_start = (0.0, 0.0, self.ball_drop_height)

        builder = newton.ModelBuilder(gravity=-9.81)

        builder.add_ground_plane(color=(0.18, 0.18, 0.2))

        deco_cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, density=0.0)
        frame_band = 0.012
        frame_bottom_z = self.cloth_height - 0.016
        frame_mesh = self.create_frame_mesh(
            outer_half_x=0.5 * self.width + frame_band,
            outer_half_y=0.5 * self.length + frame_band,
            inner_half_x=0.5 * self.width - frame_band,
            inner_half_y=0.5 * self.length - frame_band,
            top_z=self.cloth_height,
            bottom_z=frame_bottom_z,
        )
        builder.add_shape_mesh(
            body=-1,
            mesh=frame_mesh,
            cfg=deco_cfg,
            color=(0.82, 0.64, 0.24),
            label="clamp_frame",
        )

        ball_cfg = newton.ModelBuilder.ShapeConfig()
        ball_volume = (4.0 / 3.0) * math.pi * (self.ball_radius**3)
        ball_cfg.density = self.ball_mass / ball_volume
        ball_cfg.ke = self.ball_ke
        ball_cfg.kd = self.ball_kd
        ball_cfg.mu = self.ball_mu

        self.ball_body = builder.add_body(
            xform=wp.transform(self.ball_start, wp.quat_identity()),
            label="drop_ball",
        )
        self.ball_shape = builder.add_shape_sphere(
            body=self.ball_body,
            radius=self.ball_radius,
            cfg=ball_cfg,
            color=(0.88, 0.36, 0.20),
            label="drop_ball",
        )

        builder.add_cloth_grid(
            pos=wp.vec3(*self.cloth_origin),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            dim_x=self.cloth_cells_x,
            dim_y=self.cloth_cells_y,
            cell_x=self.cell_x,
            cell_y=self.cell_y,
            mass=self.cloth_particle_mass,
            fix_left=True,
            fix_right=True,
            fix_top=True,
            fix_bottom=True,
            tri_ke=self.cloth_tri_ke,
            tri_ka=self.cloth_tri_ka,
            tri_kd=self.cloth_tri_kd,
            edge_ke=self.cloth_edge_ke,
            edge_kd=self.cloth_edge_kd,
            particle_radius=self.cloth_particle_radius,
        )
        builder.color(include_bending=True)

        self.model = builder.finalize()
        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.soft_contact_mu

        self.sensor = SensorTactileArray(
            model=self.model,
            sensing_particles=list(range((self.cloth_cells_y + 1) * (self.cloth_cells_x + 1))),
            sensing_rows=self.cloth_cells_y + 1,
            sensing_cols=self.cloth_cells_x + 1,
            taxel_rows=self.taxel_rows,
            taxel_cols=self.taxel_cols,
            counterpart_shapes=[self.ball_shape],
            contact_margin=self.soft_contact_margin,
        )

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.solver_iterations,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=True,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_margin=self.soft_contact_margin)
        self.contacts = self.model.contacts(collision_pipeline=self.collision_pipeline)

        self.peak_taxel_force = 0.0
        self.peak_taxel_shear_force = 0.0
        self.peak_taxel_contact_count = 0.0
        self.total_normal_force = 0.0
        self.center_taxel_force = 0.0
        self.max_total_normal_force = 0.0
        self.max_soft_contact_count = 0
        self._taxel_count = self.taxel_rows * self.taxel_cols
        self._taxel_corner_indices = np.zeros((self._taxel_count, 4), dtype=np.int32)
        self._taxel_corner_weights = np.zeros((self._taxel_count, 4), dtype=np.float32)
        for row in range(self.taxel_rows):
            for col in range(self.taxel_cols):
                taxel_index = row * self.taxel_cols + col
                fx = (row + 0.5) * (self.cloth_rows / self.taxel_rows) - 0.5
                fy = (col + 0.5) * (self.cloth_cols / self.taxel_cols) - 0.5

                grid_row = np.clip(fx, 0.0, self.cloth_rows - 1)
                grid_col = np.clip(fy, 0.0, self.cloth_cols - 1)

                row0 = int(np.floor(grid_row))
                col0 = int(np.floor(grid_col))
                row1 = min(row0 + 1, self.cloth_rows - 1)
                col1 = min(col0 + 1, self.cloth_cols - 1)
                tx = float(grid_row - row0)
                ty = float(grid_col - col0)

                self._taxel_corner_indices[taxel_index, :] = (
                    row0 * self.cloth_cols + col0,
                    row0 * self.cloth_cols + col1,
                    row1 * self.cloth_cols + col0,
                    row1 * self.cloth_cols + col1,
                )
                self._taxel_corner_weights[taxel_index, :] = (
                    (1.0 - tx) * (1.0 - ty),
                    (1.0 - tx) * ty,
                    tx * (1.0 - ty),
                    tx * ty,
                )

        self._taxel_points_host = np.zeros((self._taxel_count, 3), dtype=np.float32)
        self._taxel_radii_host = np.zeros(self._taxel_count, dtype=np.float32)
        self._taxel_colors_host = np.zeros((self._taxel_count, 3), dtype=np.float32)
        self._taxel_line_starts_host = np.zeros((self._taxel_count, 3), dtype=np.float32)
        self._taxel_line_ends_host = np.zeros((self._taxel_count, 3), dtype=np.float32)
        self._taxel_line_colors_host = np.zeros((self._taxel_count, 3), dtype=np.float32)

        self._taxel_points = wp.zeros(self._taxel_count, dtype=wp.vec3, device=self.model.device)
        self._taxel_radii = wp.zeros(self._taxel_count, dtype=wp.float32, device=self.model.device)
        self._taxel_colors = wp.zeros(self._taxel_count, dtype=wp.vec3, device=self.model.device)
        self._taxel_line_starts = wp.zeros(self._taxel_count, dtype=wp.vec3, device=self.model.device)
        self._taxel_line_ends = wp.zeros(self._taxel_count, dtype=wp.vec3, device=self.model.device)
        self._taxel_line_colors = wp.zeros(self._taxel_count, dtype=wp.vec3, device=self.model.device)

        self.viewer.set_model(self.model)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            camera_pos = np.array([0.0, -0.7, 0.8], dtype=np.float32)
            camera_target = np.array([0.0, 0.0, self.cloth_height], dtype=np.float32)
            camera_pitch, camera_yaw = self.camera_angles_from_target(camera_pos, camera_target)
            self.viewer.set_camera(
                pos=wp.vec3(*camera_pos),
                pitch=camera_pitch,
                yaw=camera_yaw,
            )

        self.capture()

    def capture(self):
        self.graph = None

        if not wp.get_device().is_cuda:
            return

        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

            self.model.collide(self.state_0, self.contacts)
            self.solver.update_contacts(self.contacts, self.state_0)
        self.sensor.update(self.state_0, self.contacts, self.sim_dt)

    def _refresh_tactile_diagnostics(self):
        self.sensor._refresh_contact_statistics()
        normal_force_map = self.sensor.normal_force_map.numpy()
        shear_force_map = self.sensor.shear_force_map.numpy()
        contact_count_map = self.sensor.contact_count_map.numpy()
        self.peak_taxel_force = float(np.max(normal_force_map)) if normal_force_map.size else 0.0
        self.peak_taxel_shear_force = float(np.max(shear_force_map)) if shear_force_map.size else 0.0
        self.peak_taxel_contact_count = float(np.max(contact_count_map)) if contact_count_map.size else 0.0
        self.total_normal_force = float(np.sum(normal_force_map)) if normal_force_map.size else 0.0
        self.center_taxel_force = (
            float(normal_force_map[self.taxel_rows // 2, self.taxel_cols // 2]) if normal_force_map.size else 0.0
        )
        self.max_total_normal_force = max(self.max_total_normal_force, self.total_normal_force)
        self.max_soft_contact_count = max(self.max_soft_contact_count, int(self.contacts.soft_contact_count.numpy()[0]))
        if not self._viewer_is_null:
            self._update_tactile_overlay(normal_force_map)

    def _update_tactile_overlay(self, normal_force_map: np.ndarray):
        peak_force = float(np.max(normal_force_map)) if normal_force_map.size else 0.0

        particle_q = self.state_0.particle_q.numpy()

        for row in range(self.taxel_rows):
            for col in range(self.taxel_cols):
                taxel_index = row * self.taxel_cols + col
                center = np.einsum(
                    "i,ij->j",
                    self._taxel_corner_weights[taxel_index],
                    particle_q[self._taxel_corner_indices[taxel_index]],
                    optimize=True,
                ).astype(np.float32, copy=False)
                value = float(normal_force_map[row, col])
                ratio = value / peak_force if peak_force > 0.0 else 0.0

                lift = 0.004 + 0.03 * ratio
                radius = 0.003 + 0.006 * ratio
                color = np.array((ratio, 0.15 + 0.55 * (1.0 - ratio), 1.0 - ratio), dtype=np.float32)

                start = np.asarray(center + np.array([0.0, 0.0, 0.002], dtype=np.float32), dtype=np.float32)
                end = np.asarray(center + np.array([0.0, 0.0, lift], dtype=np.float32), dtype=np.float32)

                self._taxel_points_host[taxel_index] = end
                self._taxel_radii_host[taxel_index] = radius
                self._taxel_colors_host[taxel_index] = color
                self._taxel_line_starts_host[taxel_index] = start
                self._taxel_line_ends_host[taxel_index] = end
                self._taxel_line_colors_host[taxel_index] = color

        self._taxel_points.assign(self._taxel_points_host)
        self._taxel_radii.assign(self._taxel_radii_host)
        self._taxel_colors.assign(self._taxel_colors_host)
        self._taxel_line_starts.assign(self._taxel_line_starts_host)
        self._taxel_line_ends.assign(self._taxel_line_ends_host)
        self._taxel_line_colors.assign(self._taxel_line_colors_host)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self._refresh_tactile_diagnostics()

        if self._viewer_is_null:
            self.sim_time += self.frame_dt
            return

        self.viewer.log_scalar("Tactile Total Normal Force", self.total_normal_force, smoothing=6)
        self.viewer.log_scalar("Tactile Peak Normal Force", self.peak_taxel_force, smoothing=10)
        self.viewer.log_scalar("Center Taxel Normal Force", self.center_taxel_force, smoothing=8)
        self.viewer.log_scalar("Tactile Peak Shear Force", self.peak_taxel_shear_force, smoothing=10)
        self.viewer.log_scalar("Tactile Official Coverage", self.sensor.official_coverage_ratio, smoothing=10)
        self.viewer.log_array("Tactile Normal Force Heatmap", self.sensor.normal_force_map)
        self.sim_time += self.frame_dt

    def render(self):
        if self._viewer_is_null:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.end_frame()
            return

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.log_lines(
            "/tactile_array/spikes",
            self._taxel_line_starts,
            self._taxel_line_ends,
            self._taxel_line_colors,
            width=0.004,
        )
        self.viewer.log_points(
            "/tactile_array/taxels",
            self._taxel_points,
            radii=self._taxel_radii,
            colors=self._taxel_colors,
        )
        self.viewer.end_frame()

    def test_post_step(self):
        assert len(find_nonfinite_members(self.sensor)) == 0
        assert self.sensor.official_coverage_ratio >= 0.0
        assert self.sensor.official_coverage_ratio <= 1.0

    def test_final(self):
        self.test_post_step()
        workspace_margin_xy = max(0.08, 0.4 * max(self.cell_x, self.cell_y))
        workspace_margin_z = max(0.12, self.ball_radius + 0.05)
        min_x = float(-0.5 * self.width - workspace_margin_xy)
        min_y = float(-0.5 * self.length - workspace_margin_xy)
        min_z = 0.0
        max_x = float(0.5 * self.width + workspace_margin_xy)
        max_y = float(0.5 * self.length + workspace_margin_xy)
        max_z = float(self.ball_drop_height + workspace_margin_z)
        newton.examples.test_particle_state(
            self.state_0,
            "particles stay within a reasonable suspended tactile workspace volume",
            lambda q, qd: newton.math.vec_inside_limits(
                q,
                wp.vec3(min_x, min_y, min_z),
                wp.vec3(max_x, max_y, max_z),
            ),
        )
        assert self.max_soft_contact_count > 0
        assert self.max_total_normal_force > 0.0
        assert self.sensor.official_contact_count > 0
        assert self.sensor.official_coverage_ratio > 0.5
        assert self.sensor.force_source in (
            SensorTactileArray.ForceSource.OFFICIAL,
            SensorTactileArray.ForceSource.MIXED,
        )


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    if args.record_file:
        record_path = Path(args.record_file)
        if record_path.suffix == "":
            record_path = record_path.with_suffix(".bin")
        elif record_path.suffix != ".bin":
            raise ValueError("`--record-file` must use the .bin extension, or omit the extension to auto-append .bin.")

        viewer.close()
        viewer = newton.viewer.ViewerFile(str(record_path), auto_save=False)
        print(f"Recording replay to: {record_path}")
        print(f"Recording {args.num_frames} frames using ViewerFile.")

        example = Example(viewer, args)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        viewer.close()
        print(f"Replay recording saved to: {record_path}")
    else:
        example = Example(viewer, args)
        newton.examples.run(example, args)