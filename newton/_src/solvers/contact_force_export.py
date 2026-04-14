# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np

from ..geometry import ParticleFlags


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_xyz = q[:3]
    q_w = float(q[3])
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def transform_point_np(xform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return xform[:3] + _quat_rotate_np(xform[3:7], point)


def transform_vector_np(xform: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return _quat_rotate_np(xform[3:7], vec)


def fill_vbd_like_soft_contact_force_rows(
    force_rows: np.ndarray,
    rigid_contact_count: int,
    *,
    particle_q: np.ndarray,
    particle_q_prev: np.ndarray,
    particle_radius: np.ndarray,
    shape_body: np.ndarray,
    shape_material_mu: np.ndarray,
    body_q: np.ndarray | None,
    body_q_prev: np.ndarray | None,
    body_qd: np.ndarray | None,
    body_com: np.ndarray | None,
    contact_particle: np.ndarray,
    contact_shape: np.ndarray,
    contact_body_pos: np.ndarray,
    contact_body_vel: np.ndarray,
    contact_normal: np.ndarray,
    contact_ke: np.ndarray,
    contact_kd: np.ndarray,
    contact_mu: np.ndarray,
    soft_contact_margin: float,
    friction_epsilon: float,
    dt: float,
) -> None:
    """Fill ``Contacts.force`` soft rows using the VBD/Style3D soft-contact model."""
    soft_count = len(contact_particle)
    eps_u = friction_epsilon * dt

    for contact_idx in range(soft_count):
        particle_idx = int(contact_particle[contact_idx])
        shape_idx = int(contact_shape[contact_idx])
        normal = np.asarray(contact_normal[contact_idx], dtype=np.float32)
        normal_norm = float(np.linalg.norm(normal))
        if particle_idx < 0 or shape_idx < 0 or normal_norm <= 0.0:
            continue
        normal = normal / normal_norm

        body_idx = int(shape_body[shape_idx])
        body_pos_world = np.asarray(contact_body_pos[contact_idx], dtype=np.float32)
        body_vel_world = np.asarray(contact_body_vel[contact_idx], dtype=np.float32)

        if body_idx >= 0 and body_q is not None:
            xform = body_q[body_idx]
            body_pos_world = transform_point_np(xform, body_pos_world).astype(np.float32, copy=False)

            if body_q_prev is not None:
                body_pos_prev = transform_point_np(body_q_prev[body_idx], contact_body_pos[contact_idx])
                body_vel_world = ((body_pos_world - body_pos_prev) / dt).astype(np.float32, copy=False)
                body_vel_world += transform_vector_np(xform, contact_body_vel[contact_idx]).astype(np.float32, copy=False)
            else:
                body_vel_world = transform_vector_np(xform, contact_body_vel[contact_idx]).astype(np.float32, copy=False)
                if body_qd is not None and body_com is not None:
                    com_world = transform_point_np(xform, body_com[body_idx]).astype(np.float32, copy=False)
                    point_offset = body_pos_world - com_world
                    linear_vel = np.asarray(body_qd[body_idx][:3], dtype=np.float32)
                    angular_vel = np.asarray(body_qd[body_idx][3:], dtype=np.float32)
                    body_vel_world = body_vel_world + linear_vel + np.cross(angular_vel, point_offset)

        particle_pos = np.asarray(particle_q[particle_idx], dtype=np.float32)
        particle_prev_pos = np.asarray(particle_q_prev[particle_idx], dtype=np.float32)
        penetration_depth = -(
            float(np.dot(normal, particle_pos - body_pos_world))
            - float(particle_radius[particle_idx])
            - float(soft_contact_margin)
        )
        if penetration_depth <= 0.0:
            continue

        normal_load = penetration_depth * float(contact_ke[contact_idx])
        force_on_particle = normal * normal_load
        particle_dx = particle_pos - particle_prev_pos
        normal_dx = float(np.dot(normal, particle_dx))
        if normal_dx < 0.0:
            damping_coeff = float(contact_kd[contact_idx]) * float(contact_ke[contact_idx])
            force_on_particle = force_on_particle - ((damping_coeff / dt) * normal_dx) * normal

        tangential_translation = particle_dx - body_vel_world * dt
        tangential_translation = tangential_translation - normal * float(np.dot(normal, tangential_translation))
        tangential_norm = float(np.linalg.norm(tangential_translation))
        if tangential_norm > 0.0:
            friction_mu = max(float(contact_mu[contact_idx]), 0.0)
            if tangential_norm > eps_u:
                scale = friction_mu * normal_load / tangential_norm
            else:
                scale = friction_mu * normal_load * ((-tangential_norm / eps_u + 2.0) / eps_u)
            force_on_particle = force_on_particle - scale * tangential_translation

        force_on_body = -force_on_particle
        row = rigid_contact_count + contact_idx
        force_rows[row, :3] = force_on_body
        if body_idx >= 0 and body_q is not None and body_com is not None:
            com_world = transform_point_np(body_q[body_idx], body_com[body_idx]).astype(np.float32, copy=False)
            force_rows[row, 3:] = np.cross(body_pos_world - com_world, force_on_body)


def fill_semi_implicit_soft_contact_force_rows(
    force_rows: np.ndarray,
    rigid_contact_count: int,
    *,
    particle_q: np.ndarray,
    particle_qd: np.ndarray,
    particle_radius: np.ndarray,
    particle_flags: np.ndarray,
    body_q: np.ndarray | None,
    body_qd: np.ndarray | None,
    body_com: np.ndarray | None,
    shape_body: np.ndarray,
    shape_material_ke: np.ndarray,
    shape_material_kd: np.ndarray,
    shape_material_kf: np.ndarray,
    shape_material_mu: np.ndarray,
    shape_material_ka: np.ndarray,
    particle_ke: float,
    particle_kd: float,
    particle_kf: float,
    particle_mu: float,
    particle_ka: float,
    contact_particle: np.ndarray,
    contact_shape: np.ndarray,
    contact_body_pos: np.ndarray,
    contact_body_vel: np.ndarray,
    contact_normal: np.ndarray,
) -> None:
    """Fill ``Contacts.force`` soft rows using the semi-implicit soft-contact model."""
    soft_count = len(contact_particle)

    for contact_idx in range(soft_count):
        particle_idx = int(contact_particle[contact_idx])
        shape_idx = int(contact_shape[contact_idx])
        if particle_idx < 0 or shape_idx < 0:
            continue
        if (int(particle_flags[particle_idx]) & int(ParticleFlags.ACTIVE)) == 0:
            continue

        px = np.asarray(particle_q[particle_idx], dtype=np.float32)
        pv = np.asarray(particle_qd[particle_idx], dtype=np.float32)
        normal = np.asarray(contact_normal[contact_idx], dtype=np.float32)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 0.0:
            continue
        normal = normal / normal_norm

        body_idx = int(shape_body[shape_idx])
        bx = np.asarray(contact_body_pos[contact_idx], dtype=np.float32)
        body_velocity = np.asarray(contact_body_vel[contact_idx], dtype=np.float32)
        com_world = np.zeros(3, dtype=np.float32)
        angular_velocity = np.zeros(3, dtype=np.float32)
        linear_velocity = np.zeros(3, dtype=np.float32)

        if body_idx >= 0 and body_q is not None:
            xform = body_q[body_idx]
            bx = transform_point_np(xform, bx).astype(np.float32, copy=False)
            body_velocity = transform_vector_np(xform, body_velocity).astype(np.float32, copy=False)
            if body_com is not None:
                com_world = transform_point_np(xform, body_com[body_idx]).astype(np.float32, copy=False)
            if body_qd is not None:
                linear_velocity = np.asarray(body_qd[body_idx][:3], dtype=np.float32)
                angular_velocity = np.asarray(body_qd[body_idx][3:], dtype=np.float32)

        r = bx - com_world
        c = float(np.dot(normal, px - bx) - float(particle_radius[particle_idx]))
        if c > particle_ka or c >= 0.0:
            continue

        ke = 0.5 * (particle_ke + float(shape_material_ke[shape_idx]))
        kd = 0.5 * (particle_kd + float(shape_material_kd[shape_idx]))
        kf = 0.5 * (particle_kf + float(shape_material_kf[shape_idx]))
        mu = 0.5 * (particle_mu + float(shape_material_mu[shape_idx]))

        bv = linear_velocity + body_velocity + np.cross(angular_velocity, r)
        relative_velocity = pv - bv
        vn = float(np.dot(normal, relative_velocity))
        vt = relative_velocity - normal * vn

        fn = normal * c * ke
        fd = normal * min(vn, 0.0) * kd
        vt_norm = float(np.linalg.norm(vt))
        if vt_norm > 0.0:
            ft = (vt / vt_norm) * min(kf * vt_norm, abs(mu * c * ke))
        else:
            ft = np.zeros(3, dtype=np.float32)

        force_on_body = (fn + fd + ft).astype(np.float32, copy=False)
        row = rigid_contact_count + contact_idx
        force_rows[row, :3] = force_on_body
        if body_idx >= 0:
            force_rows[row, 3:] = np.cross(r, force_on_body)