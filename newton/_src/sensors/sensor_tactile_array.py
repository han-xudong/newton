# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from enum import IntEnum

import numpy as np
import warp as wp

from ..math.spatial import velocity_at_point
from ..sim import Contacts, Model, State
from ..utils.selection import match_labels


def _check_particle_index_bounds(indices: list[int], count: int, param_name: str) -> None:
    for idx in indices:
        if idx < 0 or idx >= count:
            raise IndexError(f"{param_name} contains index {idx}, but model only has {count} particles")


@wp.kernel(enable_backward=False)
def accumulate_official_tactile_forces_kernel(
    rigid_contact_count: wp.array[wp.int32],
    soft_contact_particle: wp.array[wp.int32],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_normal: wp.array[wp.vec3],
    contact_force: wp.array[wp.spatial_vector],
    particle_to_grid: wp.array[wp.int32],
    counterpart_shape_mask: wp.array[wp.int32],
    grid_taxel_y: wp.array[wp.int32],
    grid_taxel_x: wp.array[wp.int32],
    grid_taxel_weight: wp.array[float],
    # output
    normal_force_map: wp.array2d[float],
    shear_force_map: wp.array2d[float],
    contact_count_map: wp.array2d[float],
    force_map: wp.array2d[wp.vec3],
    official_contact_used: wp.array[wp.int32],
    official_contact_count: wp.array[wp.int32],
):
    tid = wp.tid()

    particle_index = soft_contact_particle[tid]
    if particle_index < 0:
        return

    shape_index = soft_contact_shape[tid]
    if shape_index < 0 or counterpart_shape_mask[shape_index] == 0:
        return

    grid_index = particle_to_grid[particle_index]
    if grid_index < 0:
        return

    force_index = rigid_contact_count[0] + tid
    total_force = -wp.spatial_top(contact_force[force_index])
    if wp.length_sq(total_force) <= 0.0:
        return

    normal = soft_contact_normal[tid]
    normal_len_sq = wp.length_sq(normal)
    if normal_len_sq <= 0.0:
        return
    if wp.abs(normal_len_sq - 1.0) > 1.0e-4:
        normal = wp.normalize(normal)

    normal_component = wp.dot(total_force, normal)
    tangential_force = total_force - normal * normal_component
    shear_magnitude = wp.length(tangential_force)

    base = grid_index * 4
    for offset in range(4):
        weight = grid_taxel_weight[base + offset]
        if weight <= 0.0:
            continue
        taxel_y = grid_taxel_y[base + offset]
        taxel_x = grid_taxel_x[base + offset]
        wp.atomic_add(normal_force_map, taxel_y, taxel_x, weight * wp.max(normal_component, 0.0))
        wp.atomic_add(shear_force_map, taxel_y, taxel_x, weight * shear_magnitude)
        wp.atomic_add(contact_count_map, taxel_y, taxel_x, weight)
        wp.atomic_add(force_map, taxel_y, taxel_x, weight * total_force)

    official_contact_used[tid] = 1
    wp.atomic_add(official_contact_count, 0, 1)


@wp.kernel(enable_backward=False)
def accumulate_estimated_tactile_forces_kernel(
    soft_contact_particle: wp.array[wp.int32],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    official_contact_used: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    prev_particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    shape_material_mu: wp.array[float],
    shape_body: wp.array[wp.int32],
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    particle_to_grid: wp.array[wp.int32],
    counterpart_shape_mask: wp.array[wp.int32],
    grid_taxel_y: wp.array[wp.int32],
    grid_taxel_x: wp.array[wp.int32],
    grid_taxel_weight: wp.array[float],
    stiffness: float,
    damping_coeff: float,
    friction_mu_base: float,
    contact_margin: float,
    eps_u: float,
    dt: float,
    # output
    normal_force_map: wp.array2d[float],
    shear_force_map: wp.array2d[float],
    contact_count_map: wp.array2d[float],
    force_map: wp.array2d[wp.vec3],
    estimated_contact_count: wp.array[wp.int32],
):
    tid = wp.tid()

    if official_contact_used[tid] != 0:
        return

    particle_index = soft_contact_particle[tid]
    if particle_index < 0:
        return

    shape_index = soft_contact_shape[tid]
    if shape_index < 0 or counterpart_shape_mask[shape_index] == 0:
        return

    grid_index = particle_to_grid[particle_index]
    if grid_index < 0:
        return

    normal = soft_contact_normal[tid]
    normal_len_sq = wp.length_sq(normal)
    if normal_len_sq <= 0.0:
        return
    if wp.abs(normal_len_sq - 1.0) > 1.0e-4:
        normal = wp.normalize(normal)

    particle_pos = particle_q[particle_index]
    particle_prev_pos = prev_particle_q[particle_index]
    particle_dx = particle_pos - particle_prev_pos

    body_pos_world = soft_contact_body_pos[tid]
    body_vel_world = soft_contact_body_vel[tid]

    body_index = shape_body[shape_index]
    if body_index >= 0:
        xform = body_q[body_index]
        body_pos_world = wp.transform_point(xform, body_pos_world)
        body_vel_world = wp.transform_vector(xform, body_vel_world)
        body_origin = wp.transform_point(xform, body_com[body_index])
        point_offset = body_pos_world - body_origin
        body_vel_world = body_vel_world + velocity_at_point(body_qd[body_index], point_offset)

    penetration_depth = -(wp.dot(normal, particle_pos - body_pos_world) - particle_radius[particle_index] - contact_margin)
    if penetration_depth <= 0.0:
        return

    base_normal_load = penetration_depth * stiffness
    normal_force = normal * base_normal_load

    normal_dx = wp.dot(normal, particle_dx)
    if normal_dx < 0.0:
        normal_force = normal_force - ((damping_coeff / dt) * normal_dx) * normal

    relative_translation = particle_dx - body_vel_world * dt
    tangential_translation = relative_translation - normal * wp.dot(normal, relative_translation)
    tangential_norm = wp.length(tangential_translation)

    friction_force = wp.vec3(0.0)
    if tangential_norm > 0.0:
        friction_mu = wp.sqrt(friction_mu_base * wp.max(shape_material_mu[shape_index], 0.0))
        scale = 0.0
        if tangential_norm > eps_u:
            scale = friction_mu * base_normal_load / tangential_norm
        else:
            scale = friction_mu * base_normal_load * ((-tangential_norm / eps_u + 2.0) / eps_u)
        friction_force = -scale * tangential_translation

    total_force = normal_force + friction_force
    normal_component = wp.dot(total_force, normal)
    tangential_force = total_force - normal * normal_component
    shear_magnitude = wp.length(tangential_force)

    base = grid_index * 4
    for offset in range(4):
        weight = grid_taxel_weight[base + offset]
        if weight <= 0.0:
            continue
        taxel_y = grid_taxel_y[base + offset]
        taxel_x = grid_taxel_x[base + offset]
        wp.atomic_add(normal_force_map, taxel_y, taxel_x, weight * wp.max(normal_component, 0.0))
        wp.atomic_add(shear_force_map, taxel_y, taxel_x, weight * shear_magnitude)
        wp.atomic_add(contact_count_map, taxel_y, taxel_x, weight)
        wp.atomic_add(force_map, taxel_y, taxel_x, weight * total_force)

    wp.atomic_add(estimated_contact_count, 0, 1)


class SensorTactileArray:
    """Measures soft contact loads on a particle skin and aggregates them into taxels.

    The sensing region is defined by a row-major list of particle indices. Contacts on
    those particles are mapped from the sensing grid to a coarser taxel grid using
    area-weighted bilinear aggregation. For each taxel, the sensor reports estimated normal
    and shear loads.

    When the extended contact attribute ``force`` is available, the sensor uses the
    solver-produced soft-contact force rows directly, matching the official contact-force
    pathway used by :class:`newton.sensors.SensorContact`. When those rows are not available,
    the sensor falls back to reconstructing loads from soft-contact geometry using the same
    body-particle contact model used by :class:`newton.solvers.SolverVBD`: a linear normal
    spring with optional damping and a projected isotropic Coulomb friction model.

    Args:
        model: The simulation model providing particle, shape, and body properties.
        sensing_particles: Row-major particle indices that form the tactile skin.
        sensing_rows: Number of rows in the sensing particle grid.
        sensing_cols: Number of columns in the sensing particle grid.
        taxel_rows: Number of rows in the output taxel grid.
        taxel_cols: Number of columns in the output taxel grid.
        counterpart_bodies: Body indices or label patterns to include as contacting bodies.
        counterpart_shapes: Shape indices or label patterns to include as contacting shapes.
        contact_margin: Extra soft-contact activation margin [m] added to the particle radius.
        friction_epsilon: Friction regularization distance factor. The smoothing distance is
            ``friction_epsilon * dt`` [m].
        verbose: If True, print initialization details. If None, uses ``wp.config.verbose``.
        request_contact_attributes: If True (default), transparently request the extended
            contact attribute ``force`` from the model.
    """

    normal_force_map: wp.array2d[float]
    """Area-weighted normal force [N], shape ``(taxel_rows, taxel_cols)``, dtype float."""

    shear_force_map: wp.array2d[float]
    """Area-weighted shear force magnitude [N], shape ``(taxel_rows, taxel_cols)``, dtype float."""

    contact_count_map: wp.array2d[float]
    """Soft contact counts [count], shape ``(taxel_rows, taxel_cols)``, dtype float."""

    force_map: wp.array2d[wp.vec3]
    """Area-weighted world-frame total contact force [N], shape ``(taxel_rows, taxel_cols)``, dtype :class:`vec3`."""

    class ForceSource(IntEnum):
        NONE = 0
        OFFICIAL = 1
        ESTIMATED = 2
        MIXED = 3

    force_source: ForceSource
    """Source of the current tactile reading, one of :class:`ForceSource`."""

    official_contact_count: int
    """Number of soft contacts that used official solver-exported force rows [count]."""

    estimated_contact_count: int
    """Number of soft contacts that fell back to geometry-based force reconstruction [count]."""

    official_coverage_ratio: float
    """Fraction of active tactile contacts that used official solver-exported force rows [0-1], dtype float."""

    sensing_particles: list[int]
    """Row-major sensing particle indices backing the tactile skin."""

    counterpart_shapes: list[int]
    """Shape indices considered valid tactile counterparts."""

    def __init__(
        self,
        model: Model,
        sensing_particles: list[int],
        sensing_rows: int,
        sensing_cols: int,
        taxel_rows: int,
        taxel_cols: int,
        *,
        counterpart_bodies: str | list[str] | list[int] | None = None,
        counterpart_shapes: str | list[str] | list[int] | None = None,
        contact_margin: float = 0.0,
        friction_epsilon: float = 1.0e-2,
        verbose: bool | None = None,
        request_contact_attributes: bool = True,
    ):
        if counterpart_bodies is not None and counterpart_shapes is not None:
            raise ValueError("At most one of `counterpart_bodies` and `counterpart_shapes` may be specified.")
        if sensing_rows <= 0 or sensing_cols <= 0:
            raise ValueError("`sensing_rows` and `sensing_cols` must be positive.")
        if taxel_rows <= 0 or taxel_cols <= 0:
            raise ValueError("`taxel_rows` and `taxel_cols` must be positive.")
        if len(sensing_particles) != sensing_rows * sensing_cols:
            raise ValueError(
                "`sensing_particles` must contain exactly `sensing_rows * sensing_cols` entries in row-major order."
            )

        particle_count = int(model.particle_count)
        _check_particle_index_bounds(sensing_particles, particle_count, "sensing_particles")
        if len(set(sensing_particles)) != len(sensing_particles):
            raise ValueError("`sensing_particles` must not contain duplicates.")

        self.device = model.device
        self.verbose = verbose if verbose is not None else wp.config.verbose
        if request_contact_attributes:
            model.request_contact_attributes("force")
        self.sensing_rows = int(sensing_rows)
        self.sensing_cols = int(sensing_cols)
        self.taxel_rows = int(taxel_rows)
        self.taxel_cols = int(taxel_cols)
        self.contact_margin = float(contact_margin)
        self.friction_epsilon = float(friction_epsilon)
        self.sensing_particles = list(sensing_particles)
        self._model = model
        self.force_source = self.ForceSource.NONE
        self.official_contact_count = 0
        self.estimated_contact_count = 0
        self.official_coverage_ratio = 0.0

        self._particle_to_grid = np.full(particle_count, -1, dtype=np.int32)
        self._particle_to_grid[np.asarray(sensing_particles, dtype=np.int32)] = np.arange(len(sensing_particles), dtype=np.int32)
        self._particle_to_grid_wp = wp.array(self._particle_to_grid, dtype=wp.int32, device=self.device)

        if counterpart_bodies is not None:
            bodies = match_labels(model.body_label, counterpart_bodies)
            body_count = len(model.body_label)
            for body_idx in bodies:
                if body_idx < 0 or body_idx >= body_count:
                    raise IndexError(f"counterpart_bodies contains index {body_idx}, but model only has {body_count} bodies")
            shape_body = model.shape_body.numpy()
            shape_indices = np.where(np.isin(shape_body, np.asarray(bodies, dtype=np.int32)))[0].tolist()
        elif counterpart_shapes is not None:
            shape_indices = match_labels(model.shape_label, counterpart_shapes)
            shape_count = model.shape_count
            for shape_idx in shape_indices:
                if shape_idx < 0 or shape_idx >= shape_count:
                    raise IndexError(
                        f"counterpart_shapes contains index {shape_idx}, but model only has {shape_count} shapes"
                    )
        else:
            shape_indices = list(range(model.shape_count))

        if not shape_indices:
            raise ValueError("No counterpart shapes matched the provided filter.")

        self.counterpart_shapes = sorted(shape_indices)
        self._counterpart_shape_mask = np.zeros(model.shape_count, dtype=bool)
        self._counterpart_shape_mask[self.counterpart_shapes] = True
        counterpart_shape_mask_int = self._counterpart_shape_mask.astype(np.int32, copy=False)
        self._counterpart_shape_mask_wp = wp.array(counterpart_shape_mask_int, dtype=wp.int32, device=self.device)

        grid_count = self.sensing_rows * self.sensing_cols
        grid_taxel_y = np.full(grid_count * 4, -1, dtype=np.int32)
        grid_taxel_x = np.full(grid_count * 4, -1, dtype=np.int32)
        grid_taxel_weight = np.zeros(grid_count * 4, dtype=np.float32)
        for grid_index in range(grid_count):
            for offset, (taxel_y, taxel_x, weight) in enumerate(self._compute_taxel_weights(grid_index)):
                flat_index = grid_index * 4 + offset
                grid_taxel_y[flat_index] = taxel_y
                grid_taxel_x[flat_index] = taxel_x
                grid_taxel_weight[flat_index] = weight
        self._grid_taxel_y = grid_taxel_y
        self._grid_taxel_x = grid_taxel_x
        self._grid_taxel_weight = grid_taxel_weight
        self._grid_taxel_y_wp = wp.array(grid_taxel_y, dtype=wp.int32, device=self.device)
        self._grid_taxel_x_wp = wp.array(grid_taxel_x, dtype=wp.int32, device=self.device)
        self._grid_taxel_weight_wp = wp.array(grid_taxel_weight, dtype=float, device=self.device)

        self._official_contact_used: wp.array[wp.int32] | None = None
        self._official_contact_count_buffer = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._estimated_contact_count_buffer = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._official_contact_capacity = 0
        self._prev_particle_positions_wp: wp.array[wp.vec3] | None = None
        self._empty_body_com = wp.zeros(0, dtype=wp.vec3, device=self.device)
        self._empty_body_q = wp.zeros(0, dtype=wp.transform, device=self.device)
        self._empty_body_qd = wp.zeros(0, dtype=wp.spatial_vector, device=self.device)

        self.normal_force_map = wp.zeros((taxel_rows, taxel_cols), dtype=float, device=self.device)
        self.shear_force_map = wp.zeros((taxel_rows, taxel_cols), dtype=float, device=self.device)
        self.contact_count_map = wp.zeros((taxel_rows, taxel_cols), dtype=float, device=self.device)
        self.force_map = wp.zeros((taxel_rows, taxel_cols), dtype=wp.vec3, device=self.device)

        if self.verbose:
            print("SensorTactileArray initialized:")
            print(f"  Sensing particles: {len(self.sensing_particles)}")
            print(f"  Sensing grid: {self.sensing_rows} x {self.sensing_cols}")
            print(f"  Taxel grid: {self.taxel_rows} x {self.taxel_cols}")
            print(f"  Counterpart shapes: {len(self.counterpart_shapes)}")

    def _ensure_runtime_buffers(self, soft_contact_capacity: int) -> None:
        if soft_contact_capacity <= self._official_contact_capacity:
            return
        self._official_contact_used = wp.zeros(soft_contact_capacity, dtype=wp.int32, device=self.device)
        self._official_contact_capacity = soft_contact_capacity

    def _taxel_weights(self, grid_index: int) -> list[tuple[int, int, float]]:
        base = grid_index * 4
        return [
            (int(self._grid_taxel_y[base + offset]), int(self._grid_taxel_x[base + offset]), weight)
            for offset in range(4)
            if (weight := float(self._grid_taxel_weight[base + offset])) > 0.0
        ]

    def _compute_taxel_weights(self, grid_index: int) -> list[tuple[int, int, float]]:
        grid_x = grid_index % self.sensing_cols
        grid_y = grid_index // self.sensing_cols

        fx = np.clip(((grid_x + 0.5) * self.taxel_cols / self.sensing_cols) - 0.5, 0.0, self.taxel_cols - 1)
        fy = np.clip(((grid_y + 0.5) * self.taxel_rows / self.sensing_rows) - 0.5, 0.0, self.taxel_rows - 1)

        x0 = int(math.floor(float(fx)))
        y0 = int(math.floor(float(fy)))
        x1 = min(x0 + 1, self.taxel_cols - 1)
        y1 = min(y0 + 1, self.taxel_rows - 1)
        tx = float(fx - x0)
        ty = float(fy - y0)

        weights: dict[tuple[int, int], float] = {}
        for taxel_y, wy in ((y0, 1.0 - ty), (y1, ty)):
            for taxel_x, wx in ((x0, 1.0 - tx), (x1, tx)):
                weights[(taxel_y, taxel_x)] = weights.get((taxel_y, taxel_x), 0.0) + (wx * wy)

        return [(taxel_y, taxel_x, weight) for (taxel_y, taxel_x), weight in weights.items() if weight > 0.0]

    def _refresh_contact_statistics(self) -> None:
        official_contact_count = int(self._official_contact_count_buffer.numpy()[0])
        estimated_contact_count = int(self._estimated_contact_count_buffer.numpy()[0])

        self.official_contact_count = official_contact_count
        self.estimated_contact_count = estimated_contact_count

        total_contact_count = official_contact_count + estimated_contact_count
        self.official_coverage_ratio = (
            float(official_contact_count) / float(total_contact_count) if total_contact_count > 0 else 0.0
        )
        if official_contact_count > 0 and estimated_contact_count > 0:
            self.force_source = self.ForceSource.MIXED
        elif official_contact_count > 0:
            self.force_source = self.ForceSource.OFFICIAL
        elif estimated_contact_count > 0:
            self.force_source = self.ForceSource.ESTIMATED
        else:
            self.force_source = self.ForceSource.NONE

    def update(self, state: State, contacts: Contacts, dt: float) -> None:
        """Update tactile readings from the current state and soft contacts.

        Args:
            state: The simulation state providing particle positions and body motion.
            contacts: The contact data to evaluate.
            dt: Time step [s] associated with this sensor update.
        """
        if dt <= 0.0:
            raise ValueError("`dt` must be positive.")
        if state is None or state.particle_q is None:
            raise ValueError("SensorTactileArray requires `state.particle_q` to be available.")
        if contacts.device != self.device:
            raise ValueError(f"Contacts device ({contacts.device}) does not match sensor device ({self.device}).")

        if self._prev_particle_positions_wp is None or len(self._prev_particle_positions_wp) != len(state.particle_q):
            self._prev_particle_positions_wp = wp.empty_like(state.particle_q)
            wp.copy(self._prev_particle_positions_wp, state.particle_q)

        self.normal_force_map.zero_()
        self.shear_force_map.zero_()
        self.contact_count_map.zero_()
        self.force_map.zero_()

        self._official_contact_count_buffer.zero_()
        self._estimated_contact_count_buffer.zero_()

        if contacts.soft_contact_max > 0:
            self._ensure_runtime_buffers(contacts.soft_contact_max)
            assert self._official_contact_used is not None
            self._official_contact_used.zero_()

            if contacts.force is not None:
                wp.launch(
                    accumulate_official_tactile_forces_kernel,
                    dim=contacts.soft_contact_max,
                    inputs=[
                        contacts.rigid_contact_count,
                        contacts.soft_contact_particle,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_normal,
                        contacts.force,
                        self._particle_to_grid_wp,
                        self._counterpart_shape_mask_wp,
                        self._grid_taxel_y_wp,
                        self._grid_taxel_x_wp,
                        self._grid_taxel_weight_wp,
                    ],
                    outputs=[
                        self.normal_force_map,
                        self.shear_force_map,
                        self.contact_count_map,
                        self.force_map,
                        self._official_contact_used,
                        self._official_contact_count_buffer,
                    ],
                    device=self.device,
                )

            stiffness = float(self._model.soft_contact_ke)
            damping_coeff = float(self._model.soft_contact_kd) * stiffness
            friction_mu_base = max(float(self._model.soft_contact_mu), 0.0)
            eps_u = self.friction_epsilon * dt

            wp.launch(
                accumulate_estimated_tactile_forces_kernel,
                dim=contacts.soft_contact_max,
                inputs=[
                    contacts.soft_contact_particle,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    self._official_contact_used,
                    state.particle_q,
                    self._prev_particle_positions_wp,
                    self._model.particle_radius,
                    self._model.shape_material_mu,
                    self._model.shape_body,
                    self._model.body_com if self._model.body_com is not None else self._empty_body_com,
                    state.body_q if state.body_q is not None else self._empty_body_q,
                    state.body_qd if state.body_qd is not None else self._empty_body_qd,
                    self._particle_to_grid_wp,
                    self._counterpart_shape_mask_wp,
                    self._grid_taxel_y_wp,
                    self._grid_taxel_x_wp,
                    self._grid_taxel_weight_wp,
                    stiffness,
                    damping_coeff,
                    friction_mu_base,
                    self.contact_margin,
                    eps_u,
                    dt,
                ],
                outputs=[
                    self.normal_force_map,
                    self.shear_force_map,
                    self.contact_count_map,
                    self.force_map,
                    self._estimated_contact_count_buffer,
                ],
                device=self.device,
            )

        if not self.device.is_capturing:
            self._refresh_contact_statistics()
        wp.copy(self._prev_particle_positions_wp, state.particle_q)