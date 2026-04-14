# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import types
import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorTactileArray
from newton.solvers import SolverSemiImplicit
from newton.solvers import SolverVBD
from newton.solvers import SolverStyle3D
from newton.solvers import style3d


def _make_cloth_model(device):
    builder = newton.ModelBuilder(gravity=0.0)
    shape = builder.add_shape_box(body=-1, hx=0.2, hy=0.2, hz=0.01, label="pad")
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=1,
        dim_y=1,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.01,
        fix_left=False,
        fix_right=False,
        particle_radius=0.05,
    )
    builder.color()
    model = builder.finalize(device=device)
    model.soft_contact_ke = 100.0
    model.soft_contact_kd = 0.1
    model.soft_contact_mu = 0.5
    return model, shape


def _make_style3d_cloth_model(device):
    builder = newton.ModelBuilder(gravity=0.0)
    SolverStyle3D.register_custom_attributes(builder)
    shape = builder.add_shape_box(body=-1, hx=0.2, hy=0.2, hz=0.01, label="pad")
    style3d.add_cloth_grid(
        builder,
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=1,
        dim_y=1,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.01,
        particle_radius=0.05,
        tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e1),
        edge_aniso_ke=wp.vec3(2.0e-4, 1.0e-4, 5.0e-5),
    )
    model = builder.finalize(device=device)
    model.soft_contact_ke = 100.0
    model.soft_contact_kd = 0.1
    model.soft_contact_mu = 0.5
    return model, builder, shape


def _assign_soft_contacts(device, contacts, particle_ids, shape_ids, body_pos, body_vel, normals):
    with wp.ScopedDevice(device):
        contacts.soft_contact_count = wp.array([len(particle_ids)], dtype=wp.int32)
        contacts.soft_contact_particle = wp.array(particle_ids, dtype=wp.int32)
        contacts.soft_contact_shape = wp.array(shape_ids, dtype=wp.int32)
        contacts.soft_contact_body_pos = wp.array(body_pos, dtype=wp.vec3)
        contacts.soft_contact_body_vel = wp.array(body_vel, dtype=wp.vec3)
        contacts.soft_contact_normal = wp.array(normals, dtype=wp.vec3)


def _assign_contact_force_rows(device, contacts, rigid_count, force_rows):
    with wp.ScopedDevice(device):
        contacts.rigid_contact_count = wp.array([rigid_count], dtype=wp.int32)
        contacts.force = wp.array(force_rows, dtype=wp.spatial_vector)


class TestSensorTactileArray(unittest.TestCase):
    def test_semi_implicit_update_contacts_populates_soft_contact_force_rows(self):
        device = wp.get_device()
        model, _shape = _make_cloth_model(device)
        model.request_contact_attributes("force")
        solver = SolverSemiImplicit(model)

        state0 = model.state()
        contacts0 = model.contacts()
        state0.particle_q.assign(
            wp.array(
                [
                    [0.0, 0.0, 0.045],
                    [0.1, 0.0, 0.060],
                    [0.0, 0.1, 0.060],
                    [0.1, 0.1, 0.048],
                ],
                dtype=wp.vec3,
                device=device,
            )
        )
        state0.particle_qd.zero_()

        model.collide(state0, contacts0)
        solver._last_dt = 0.1
        solver.update_contacts(contacts0, state0)

        rigid_count = int(contacts0.rigid_contact_count.numpy()[0])
        soft_count = int(contacts0.soft_contact_count.numpy()[0])
        force_rows = contacts0.force.numpy()

        self.assertGreaterEqual(soft_count, 1)
        self.assertGreater(float(np.linalg.norm(force_rows[rigid_count, :3])), 0.0)

    def test_vbd_update_contacts_populates_soft_contact_force_rows(self):
        device = wp.get_device()
        model, _shape = _make_cloth_model(device)
        model.request_contact_attributes("force")
        solver = SolverVBD(model)

        state0 = model.state()
        contacts0 = model.contacts()

        state0.particle_q.assign(
            wp.array(
                [
                    [0.0, 0.0, 0.045],
                    [0.1, 0.0, 0.060],
                    [0.0, 0.1, 0.060],
                    [0.1, 0.1, 0.048],
                ],
                dtype=wp.vec3,
                device=device,
            )
        )

        model.collide(state0, contacts0)
        solver._last_dt = 0.1
        solver.particle_q_prev.assign(state0.particle_q)

        solver.update_contacts(contacts0, state0)

        rigid_count = int(contacts0.rigid_contact_count.numpy()[0])
        soft_count = int(contacts0.soft_contact_count.numpy()[0])
        force_rows = contacts0.force.numpy()

        self.assertGreaterEqual(soft_count, 1)
        self.assertEqual(force_rows.shape[0], contacts0.rigid_contact_max + contacts0.soft_contact_max)
        self.assertGreater(float(np.linalg.norm(force_rows[rigid_count, :3])), 0.0)

    def test_vbd_update_contacts_populates_force_rows_inside_soft_contact_margin(self):
        device = wp.get_device()
        model, _shape = _make_cloth_model(device)
        model.request_contact_attributes("force")
        solver = SolverVBD(model)

        state0 = model.state()
        contacts0 = model.contacts()

        state0.particle_q.assign(
            wp.array(
                [
                    [0.0, 0.0, 0.065],
                    [0.1, 0.0, 0.12],
                    [0.0, 0.1, 0.12],
                    [0.1, 0.1, 0.12],
                ],
                dtype=wp.vec3,
                device=device,
            )
        )

        model.collide(state0, contacts0)
        solver._last_dt = 0.1
        solver.particle_q_prev.assign(state0.particle_q)
        solver.update_contacts(contacts0, state0)

        rigid_count = int(contacts0.rigid_contact_count.numpy()[0])
        soft_count = int(contacts0.soft_contact_count.numpy()[0])
        force_rows = contacts0.force.numpy()

        self.assertGreaterEqual(soft_count, 1)
        self.assertGreater(float(contacts0.soft_contact_margin), 0.0)
        self.assertGreater(float(np.linalg.norm(force_rows[rigid_count, :3])), 0.0)

    def test_style3d_update_contacts_populates_soft_contact_force_rows(self):
        if not wp.is_cuda_available():
            self.skipTest("Style3D contact-force export test requires CUDA")

        device = wp.get_device("cuda:0")
        model, builder, _shape = _make_style3d_cloth_model(device)
        model.request_contact_attributes("force")
        solver = SolverStyle3D(model, iterations=1, linear_iterations=2)
        solver._precompute(builder)

        state0 = model.state()
        state1 = model.state()
        contacts0 = model.contacts()
        control = model.control()

        state0.particle_q.assign(
            wp.array(
                [
                    [0.0, 0.0, 0.04],
                    [0.1, 0.0, 0.12],
                    [0.0, 0.1, 0.12],
                    [0.1, 0.1, 0.12],
                ],
                dtype=wp.vec3,
                device=device,
            )
        )
        state0.particle_qd.zero_()

        model.collide(state0, contacts0)
        self.assertGreaterEqual(int(contacts0.soft_contact_count.numpy()[0]), 1)

        solver.step(state0, state1, control, contacts0, 0.01)
        model.collide(state1, contacts0)
        solver.update_contacts(contacts0, state1)

        rigid_count = int(contacts0.rigid_contact_count.numpy()[0])
        soft_count = int(contacts0.soft_contact_count.numpy()[0])
        force_rows = contacts0.force.numpy()

        self.assertGreaterEqual(soft_count, 1)
        self.assertGreater(float(contacts0.soft_contact_margin), 0.0)
        self.assertGreater(float(np.linalg.norm(force_rows[rigid_count, :3])), 0.0)

    def test_official_soft_contact_force_rows_take_priority(self):
        device = wp.get_device()
        model, shape = _make_cloth_model(device)

        sensor = SensorTactileArray(
            model,
            sensing_particles=[0, 1, 2, 3],
            sensing_rows=2,
            sensing_cols=2,
            taxel_rows=2,
            taxel_cols=2,
            counterpart_shapes=[shape],
        )

        contacts = newton.Contacts(0, 2, device=device, requested_attributes={"force"})
        _assign_soft_contacts(
            device,
            contacts,
            [0, 3],
            [shape, shape],
            [[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        )
        _assign_contact_force_rows(
            device,
            contacts,
            0,
            [
                (0.1, 0.0, -0.3, 0.0, 0.0, 0.0),
                (0.0, 0.0, -0.15, 0.0, 0.0, 0.0),
            ],
        )

        particle_q = wp.array(
            [
                [0.0, 0.0, 0.045],
                [0.1, 0.0, 0.060],
                [0.0, 0.1, 0.060],
                [0.1, 0.1, 0.048],
            ],
            dtype=wp.vec3,
            device=device,
        )
        state = types.SimpleNamespace(particle_q=particle_q, body_q=None, body_qd=None)
        sensor.update(state, contacts, dt=0.1)

        normal_map = sensor.normal_force_map.numpy()
        shear_map = sensor.shear_force_map.numpy()
        force_map = sensor.force_map.numpy()

        self.assertAlmostEqual(float(normal_map[0, 0]), 0.3, places=6)
        self.assertAlmostEqual(float(normal_map[1, 1]), 0.15, places=6)
        self.assertAlmostEqual(float(shear_map[0, 0]), 0.1, places=6)
        self.assertAlmostEqual(float(force_map[0, 0, 0]), -0.1, places=6)
        self.assertAlmostEqual(float(force_map[0, 0, 2]), 0.3, places=6)
        self.assertEqual(sensor.force_source, SensorTactileArray.ForceSource.OFFICIAL)
        self.assertEqual(sensor.official_contact_count, 2)
        self.assertEqual(sensor.estimated_contact_count, 0)
        self.assertAlmostEqual(float(sensor.official_coverage_ratio), 1.0, places=6)

    def test_area_weighted_taxel_aggregation(self):
        device = wp.get_device()
        model, shape = _make_cloth_model(device)

        sensor = SensorTactileArray(
            model,
            sensing_particles=[0, 1, 2, 3],
            sensing_rows=2,
            sensing_cols=2,
            taxel_rows=4,
            taxel_cols=4,
            counterpart_shapes=[shape],
        )

        contacts = newton.Contacts(0, 1, device=device, requested_attributes={"force"})
        _assign_soft_contacts(
            device,
            contacts,
            [0],
            [shape],
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0]],
        )
        _assign_contact_force_rows(device, contacts, 0, [(0.0, 0.0, -1.0, 0.0, 0.0, 0.0)])

        state = types.SimpleNamespace(
            particle_q=wp.array(
                [[0.0, 0.0, 0.045], [0.1, 0.0, 0.060], [0.0, 0.1, 0.060], [0.1, 0.1, 0.048]],
                dtype=wp.vec3,
                device=device,
            ),
            body_q=None,
            body_qd=None,
        )
        sensor.update(state, contacts, dt=0.1)

        normal_map = sensor.normal_force_map.numpy()
        count_map = sensor.contact_count_map.numpy()

        self.assertAlmostEqual(float(normal_map[0, 0]), 0.25, places=6)
        self.assertAlmostEqual(float(normal_map[0, 1]), 0.25, places=6)
        self.assertAlmostEqual(float(normal_map[1, 0]), 0.25, places=6)
        self.assertAlmostEqual(float(normal_map[1, 1]), 0.25, places=6)
        self.assertAlmostEqual(float(np.sum(normal_map)), 1.0, places=6)
        self.assertAlmostEqual(float(np.sum(count_map)), 1.0, places=6)

    def test_force_source_mixed_when_official_rows_are_partial(self):
        device = wp.get_device()
        model, shape = _make_cloth_model(device)

        sensor = SensorTactileArray(
            model,
            sensing_particles=[0, 1, 2, 3],
            sensing_rows=2,
            sensing_cols=2,
            taxel_rows=2,
            taxel_cols=2,
            counterpart_shapes=[shape],
        )

        contacts = newton.Contacts(0, 2, device=device, requested_attributes={"force"})
        _assign_soft_contacts(
            device,
            contacts,
            [0, 3],
            [shape, shape],
            [[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        )
        _assign_contact_force_rows(
            device,
            contacts,
            0,
            [
                (0.0, 0.0, -0.3, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ],
        )

        state = types.SimpleNamespace(
            particle_q=wp.array(
                [[0.0, 0.0, 0.045], [0.1, 0.0, 0.060], [0.0, 0.1, 0.060], [0.1, 0.1, 0.048]],
                dtype=wp.vec3,
                device=device,
            ),
            body_q=None,
            body_qd=None,
        )
        sensor.update(state, contacts, dt=0.1)

        self.assertEqual(sensor.force_source, SensorTactileArray.ForceSource.MIXED)
        self.assertEqual(sensor.official_contact_count, 1)
        self.assertEqual(sensor.estimated_contact_count, 1)
        self.assertAlmostEqual(float(sensor.official_coverage_ratio), 0.5, places=6)

    def test_soft_contact_force_map(self):
        device = wp.get_device()
        model, shape = _make_cloth_model(device)

        sensor = SensorTactileArray(
            model,
            sensing_particles=[0, 1, 2, 3],
            sensing_rows=2,
            sensing_cols=2,
            taxel_rows=2,
            taxel_cols=2,
            counterpart_shapes=[shape],
        )

        contacts = newton.Contacts(0, 4, device=device)
        normals = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        body_pos = [[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]]
        body_vel = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        _assign_soft_contacts(device, contacts, [0, 3], [shape, shape], body_pos, body_vel, normals)

        particle_q0 = wp.array(
            [
                [0.0, 0.0, 0.045],
                [0.1, 0.0, 0.060],
                [0.0, 0.1, 0.060],
                [0.1, 0.1, 0.048],
            ],
            dtype=wp.vec3,
            device=device,
        )
        state0 = types.SimpleNamespace(particle_q=particle_q0, body_q=None, body_qd=None)
        sensor.update(state0, contacts, dt=0.1)

        normal_map0 = sensor.normal_force_map.numpy()
        shear_map0 = sensor.shear_force_map.numpy()
        count_map0 = sensor.contact_count_map.numpy()

        self.assertAlmostEqual(float(normal_map0[0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(normal_map0[1, 1]), 0.2, places=5)
        self.assertAlmostEqual(float(shear_map0[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(count_map0[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(count_map0[1, 1]), 1.0, places=6)

        particle_q1 = wp.array(
            [
                [0.002, 0.0, 0.044],
                [0.1, 0.0, 0.060],
                [0.0, 0.1, 0.060],
                [0.1, 0.1, 0.048],
            ],
            dtype=wp.vec3,
            device=device,
        )
        state1 = types.SimpleNamespace(particle_q=particle_q1, body_q=None, body_qd=None)
        sensor.update(state1, contacts, dt=0.1)

        normal_map1 = sensor.normal_force_map.numpy()
        shear_map1 = sensor.shear_force_map.numpy()
        force_map1 = sensor.force_map.numpy()
        base_normal_load = (0.05 - 0.044) * model.soft_contact_ke
        expected_shear = float(np.sqrt(model.soft_contact_mu * model.shape_material_mu.numpy()[shape]) * base_normal_load)

        self.assertAlmostEqual(float(normal_map1[0, 0]), 0.7, places=5)
        self.assertAlmostEqual(float(shear_map1[0, 0]), expected_shear, places=5)
        self.assertAlmostEqual(float(normal_map1[1, 1]), 0.2, places=5)
        self.assertAlmostEqual(float(force_map1[0, 0, 0]), -expected_shear, places=5)
        self.assertAlmostEqual(float(force_map1[0, 0, 2]), 0.7, places=5)
        self.assertEqual(sensor.force_source, SensorTactileArray.ForceSource.ESTIMATED)
        self.assertEqual(sensor.official_contact_count, 0)
        self.assertEqual(sensor.estimated_contact_count, 2)
        self.assertAlmostEqual(float(sensor.official_coverage_ratio), 0.0, places=6)

    def test_counterpart_body_filter(self):
        device = wp.get_device()
        builder = newton.ModelBuilder(gravity=0.0)
        body = builder.add_body(label="tool")
        shape_body = builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1, label="tool_shape")
        shape_static = builder.add_shape_box(body=-1, hx=0.1, hy=0.1, hz=0.1, label="ground")
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            dim_x=1,
            dim_y=1,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
            particle_radius=0.05,
        )
        model = builder.finalize(device=device)
        model.soft_contact_ke = 100.0
        model.soft_contact_kd = 0.0
        model.soft_contact_mu = 0.5

        sensor = SensorTactileArray(
            model,
            sensing_particles=[0, 1, 2, 3],
            sensing_rows=2,
            sensing_cols=2,
            taxel_rows=2,
            taxel_cols=2,
            counterpart_bodies="tool",
        )

        contacts = newton.Contacts(0, 2, device=device)
        _assign_soft_contacts(
            device,
            contacts,
            [0, 3],
            [shape_body, shape_static],
            [[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        )

        particle_q = wp.array(
            [
                [0.0, 0.0, 0.045],
                [0.1, 0.0, 0.060],
                [0.0, 0.1, 0.060],
                [0.1, 0.1, 0.045],
            ],
            dtype=wp.vec3,
            device=device,
        )
        body_q = wp.array([wp.transform((0.0, 0.0, 0.0), wp.quat_identity())], dtype=wp.transform, device=device)
        body_qd = wp.zeros((1,), dtype=wp.spatial_vector, device=device)
        state = types.SimpleNamespace(particle_q=particle_q, body_q=body_q, body_qd=body_qd)

        sensor.update(state, contacts, dt=0.1)
        count_map = sensor.contact_count_map.numpy()

        self.assertAlmostEqual(float(count_map[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(count_map[1, 1]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)