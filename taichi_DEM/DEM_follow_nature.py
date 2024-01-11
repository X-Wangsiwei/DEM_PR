import taichi as ti
# import taichi_glsl as ts
import taichi.math as tm
import math
import os
import numpy as np
import argparse
import subprocess

vec2 = tm.vec2
vec3 = tm.vec3
vec4 = tm.vec4


@ti.dataclass
class tik_system:
    tik: ti.f32
    shake_sate_machine: ti.f32


@ti.dataclass
class ParticleRotate:
    id: ti.i32  # id of particle
    pos: vec2  # position
    mass: ti.f32  # mass
    radius: ti.f32  # radius
    color: ti.u32  # color

    # Translational attributes, all in GLOBAL coordinates
    speed: ti.f32  # speed value
    vel: vec2  # velocity
    acc: vec2  # acceleration
    force: vec2  # external force

    # Rotational attributes, all in GLOBAL coordinates
    orient: ti.f32  # orientation
    inertia: ti.f32  # inertia
    ang_vel: ti.f32  # angular velocity
    ang_acc: ti.f32  # angular acceleration
    torque: ti.f32  # torque

    energy: ti.f32


@ti.dataclass
class Contact:
    i: ti.i32
    j: ti.i32
    isActive: ti.i32  # Contact exists: 1 - exist 0 - not exist
    shear_displacement: vec2  # Shear displacement stored in the contact
    position: vec2  # Position of contact point
    normal_force: vec2
    shear_force: vec2


@ti.dataclass
class Wall:
    # Wall equation: Ax + By - C = 0
    normal: vec2
    distance: float
    speed: vec2


@ti.data_oriented
class ParticleHertzMindlin:
    def __init__(self, args):
        """ python scope data """
        # size data
        self.particle_num = args.particle_num
        self.grid_n = args.grid_n
        self.grid_size = 1.0 / self.grid_n  # Simulation domain of size [0, 1]

        # env setting
        self.window_size = args.window_size
        self.gravity = args.gravity
        self.dt = args.dt
        self.action_substeps = args.action_substeps
        self.substeps = args.substeps

        self.n_walls = args.nWall
        if args.nWall > 0:
            self.wf = Wall.field(shape=args.nWall)
            self.wcf = Contact.field(shape=(args.particle_num, args.nWall))
        else:
            self.wf = Wall.field(shape=1)
            self.wcf = Contact.field(shape=(args.particle_num, 1))

        self.wall_poissonRatio = args.wall_poissonRatio
        self.wall_elasticModulus = args.wall_elasticModulus
        self.wall_fraction_coef = args.wall_fraction_coef
        self.wall_restitution_coef = args.wall_restitution_coef

        # particle property
        self.rotation = args.rotation
        self.contact_num_max = args.contact_num_max
        self.density = args.density
        self.elasticModulus = args.elasticModulus
        self.poissonRatio = args.poissonRatio
        self.fraction_coef = args.fraction_coef  # typical: 0.75
        self.restitution_coef = args.restitution_coef  # typical: 0.01
        self.r_min_orig = args.radius_min
        self.r_max_orig = args.radius_max
        self.r_min = args.radius_min / args.window_size  # typical: 0.002
        self.r_max = args.radius_max / args.window_size  # typical: 0.003
        self.r_speed = args.radius_speed / args.window_size  # typical: 0.0001
        # self.mass = 0.1

        """ taichi scope data """
        # particles
        self.particles = ParticleRotate.field(shape=(self.particle_num,))

        # grid
        self.list_head = ti.field(dtype=ti.i32, shape=self.grid_n * self.grid_n)
        self.list_cur = ti.field(dtype=ti.i32, shape=self.grid_n * self.grid_n)
        self.list_tail = ti.field(dtype=ti.i32, shape=self.grid_n * self.grid_n)

        self.particles_count = ti.field(dtype=ti.i32,
                                        shape=(self.grid_n, self.grid_n),
                                        name="particles_count")
        self.column_sum = ti.field(dtype=ti.i32, shape=self.grid_n, name="column_sum")
        self.prefix_sum = ti.field(dtype=ti.i32, shape=(self.grid_n, self.grid_n), name="prefix_sum")
        self.particle_id = ti.field(dtype=ti.i32, shape=self.particle_num, name="particle_id")

        self.cf = Contact.field(shape=self.contact_num_max * self.particle_num)  # neighbors for every particle
        self.cfn = ti.field(ti.i32, shape=self.particle_num)  # neighbor counter for every particle

        # shake settings
        self.quarter_period = 5000.0
        self.max_movement = 0.18
        self.a_shake = self.max_movement / (self.quarter_period ** 2)
        # self.shake_sate_machine = 0
        self.direction = vec2(1.0, 0.0)
        self.tik_sys = tik_system.field(shape=(1,))
        self.tik_sys[0].tik = 0
        self.tik_sys[0].shake_sate_machine = 0

        """ value init """
        self.init_func()

    @ti.kernel
    def init_func(self):
        for i in self.particles:
            self.particles[i].id = i + 1
            loc = i * 2 * self.r_min
            region_width = int(self.particle_num ** 0.5) * 2 * self.r_min
            start = [0.5 - region_width / 2, 0.5 - region_width / 2]
            pos = vec2(loc % region_width + start[0],
                       loc // region_width * 2 * self.r_min + start[1])
            self.particles[i].pos = pos
            self.particles[i].radius = self.r_min
            self.particles[i].mass = self.density * tm.pi * (self.particles[i].radius * self.window_size) ** 2
            print(f"robot {i} mass: {self.particles[i].mass}, "
                  f"fraction force: {self.particles[i].mass * self.gravity * self.fraction_coef}")
            # self.particles[i].mass = 0.1
            self.particles[i].orient = 0.0
            self.particles[i].inertia = 0.5 * self.particles[i].mass * (
                    (self.particles[i].radius * self.window_size) ** 2)
            self.particles[i].vel = vec2([0.0, 0.0])
            self.particles[i].ang_vel = 0.0

        if args.nWall == 4:
            self.wf[0].normal = vec2(1.0, 0.0)
            self.wf[0].distance = 0.01
            self.wf[0].speed = vec2(0.0, 0.0)

            self.wf[1].normal = vec2(-1.0, 0.0)
            self.wf[1].distance = -0.8
            self.wf[1].speed = vec2(0.0, 0.0)

            self.wf[2].normal = vec2(0.0, 1.0)
            self.wf[2].distance = 0.1
            self.wf[2].speed = vec2(0.0, 0.0)

            self.wf[3].normal = vec2(0.0, -1.0)
            self.wf[3].distance = -0.99
            self.wf[3].speed = vec2(0.0, 0.0)

    @ti.func
    def append_contact_offset(self, i):
        ret = -1
        offset = ti.atomic_add(self.cfn[i], 1)
        if offset < self.contact_num_max:
            ret = i * self.contact_num_max + offset
        return ret

    @ti.func
    def search_active_contact_offset(self, i, j):
        ret = -1
        for offset in range(self.cfn[i]):
            tmp = ret = i * self.contact_num_max + offset
            if self.cf[tmp].j == j and self.cf[tmp].isActive == 1:
                ret = tmp
                break
        return ret

    @ti.func
    def remove_inactive_contact(self, i):
        active_count = 0
        for j in range(self.cfn[i]):
            if self.cf[i * self.contact_num_max + j].isActive == 1:
                active_count += 1
        offset = 0
        for j in range(self.cfn[i]):
            if self.cf[i * self.contact_num_max + j].isActive == 1:
                self.cf[i * self.contact_num_max + offset] = self.cf[i * self.contact_num_max + j]
                offset += 1
                if offset >= active_count:
                    break
        for j in range(active_count, self.cfn[i]):
            self.cf[i * self.contact_num_max + j].isActive = 0
        self.cfn[i] = active_count

    @ti.kernel
    def late_clear_state(self):
        for i in self.particles:
            self.remove_inactive_contact(i)

    @ti.kernel
    def update(self):
        # self.shake_wall()
        for i in self.particles:
            acc = self.particles[i].force / self.particles[i].mass
            self.particles[i].vel += (self.particles[i].acc + acc) * self.dt / 2.0
            self.particles[i].pos += self.particles[i].vel * self.dt + 0.5 * acc * (self.dt ** 2)
            self.particles[i].acc = acc

            if self.rotation:
                ang_acc = self.particles[i].torque / self.particles[i].inertia
                self.particles[i].ang_vel += (self.particles[i].ang_acc + ang_acc) * self.dt / 2.0
                self.particles[i].orient += self.particles[i].ang_vel * self.dt + \
                                            0.5 * ang_acc * (self.dt ** 2)
                self.particles[i].ang_acc = ang_acc

                self.particles[i].inertia = 0.5 * self.particles[i].mass * \
                                            (self.particles[i].radius * self.window_size) ** 2
            self.particles[i].speed = tm.length(self.particles[i].vel)
            self.particles[i].energy = \
                - self.particles[i].pos[1] * self.particles[i].mass * self.gravity + \
                0.5 * self.particles[i].mass * self.particles[i].speed ** 2 + \
                0.5 * self.particles[i].inertia * (self.particles[i].ang_vel ** 2)

    @ti.kernel
    def apply_bc(self):
        bounce_coef = 0.3  # Velocity damping
        for i in self.particles:
            x = self.particles[i].pos[0]
            y = self.particles[i].pos[1]

            if y - self.particles[i].radius < 0:
                self.particles[i].pos[1] = self.particles[i].radius
                self.particles[i].vel[1] *= -bounce_coef
                self.particles[i].ang_vel *= bounce_coef

            elif y + self.particles[i].radius > 1.0:
                self.particles[i].pos[1] = 1.0 - self.particles[i].radius
                self.particles[i].vel[1] *= -bounce_coef
                self.particles[i].ang_vel *= bounce_coef

            if x - self.particles[i].radius < 0:
                self.particles[i].pos[0] = self.particles[i].radius
                self.particles[i].vel[0] *= -bounce_coef
                self.particles[i].ang_vel *= bounce_coef

            elif x + self.particles[i].radius > 1.0:
                self.particles[i].pos[0] = 1.0 - self.particles[i].radius
                self.particles[i].vel[0] *= -bounce_coef
                self.particles[i].ang_vel *= bounce_coef

    @ti.func
    def resolve(self, i, j):
        offset = self.search_active_contact_offset(i, j)

        rel_pos = self.particles[j].pos - self.particles[i].pos
        rel_vel = self.particles[j].vel - self.particles[i].vel
        # dist = tm.sqrt(rel_pos[0] ** 2 + rel_pos[1] ** 2)
        dist = tm.length(rel_pos)
        delta = - dist + (self.particles[i].radius + self.particles[j].radius)
        if delta < 0:
            if offset >= 0:
                self.cf[offset].isActive = 0
                self.cf[offset].shear_displacement = vec2(0.0, 0.0)
                self.cf[offset].position = vec2(0.0, 0.0)
                self.cf[offset].normal_force = vec2(0.0, 0.0)
                self.cf[offset].shear_force = vec2(0.0, 0.0)
            # external force
            normal = rel_pos / dist
            Fn = normal
            p1, p2 = 0.045 * self.r_min, 0.09 * self.r_min
            # f1, f2 = 3.5, 3.0*0.000015884
            f1 = 3 * self.fraction_coef * self.particles[0].mass * self.gravity
            f2 = p2 * p2 * 4 * self.fraction_coef * self.particles[0].mass * self.gravity
            if delta > - p1:
                Fn *= f1
            elif delta > - p2:
                Fn *= (f1 + (f2 / (p2**2) - 2.5) / (p2 - p1) * (-delta - p2))
            else:
                Fn *= f2 / (delta ** 2)
            ti.atomic_add(self.particles[i].force, Fn)
            ti.atomic_add(self.particles[j].force, - Fn)
        else:  # contact
            if offset < 0:
                offset = self.append_contact_offset(i)
                if offset < 0:
                    print(f"ERROR: coordinate number > "
                          f"set_max_coordinate_number({self.contact_num_max})")
                self.cf[offset] = Contact(i=i, j=j, shear_displacement=vec2(0.0, 0.0))

            # get normal and tangent direction
            normal = rel_pos / dist
            tangent = vec2(-normal[1], normal[0])

            # get contact point and distance to core
            c_pos = self.particles[i].pos + normal * (self.particles[i].radius - 0.5 * delta)
            self.cf[offset].position = c_pos

            rel_vn = (rel_vel.dot(normal)) * normal
            rel_vt = rel_vel - rel_vn

            beta = (1.0 / tm.sqrt(1.0 + (tm.pi / ti.log(self.restitution_coef)) ** 2))
            e_star = 2.0 * self.elasticModulus / (1.0 - self.poissonRatio ** 2)
            g_star = self.elasticModulus / ((2.0 - self.poissonRatio) * (1.0 + self.poissonRatio))
            r_star = 1.0 / (1.0 / self.particles[i].radius + 1.0 / self.particles[j].radius)
            m_star = 1.0 / (1.0 / self.particles[i].mass + 1.0 / self.particles[j].mass)

            kn = 4.0 / 3.0 * e_star * tm.sqrt(r_star * delta)
            cn = 2.0 * beta * tm.sqrt(5.0 / 6.0 * kn * m_star)
            Fn = kn * delta * normal - cn * rel_vn + 2.5 * normal

            ks = 8 * g_star * tm.sqrt(r_star * delta)
            cs = 2.0 * beta * tm.sqrt(5.0 / 6.0 * ks * m_star)
            shear_increment = rel_vt * self.dt
            self.cf[offset].shear_displacement += shear_increment
            Fs_try = ks * tm.dot(self.cf[offset].shear_displacement, tangent) * tangent
            Fs = Fs_try - cs * rel_vt

            Fn_abs = tm.length(Fn)
            Fs_abs = tm.length(Fs)

            if Fs_abs > self.fraction_coef * Fn_abs:
                Fs = (self.fraction_coef * Fn_abs) * Fs / Fs_abs
                self.cf[offset].shear_displacement[0] = Fs[0] / ks
                self.cf[offset].shear_displacement[1] = Fs[1] / ks

            ti.atomic_add(self.particles[i].force, -(Fn + Fs))
            ti.atomic_add(self.particles[j].force, Fn + Fs)

            self.cf[offset].normal_force = Fn
            self.cf[offset].shear_force = Fs

    @ti.func
    def evaluate_wall(self, i, j):
        dis = tm.dot(self.particles[i].pos, self.wf[j].normal) - self.wf[j].distance
        gap = dis - self.particles[i].radius
        if gap < 0:
            self.particles[i].vel += self.wf[j].speed * self.direction
            delta_n = ti.abs(gap)
            r_i = - dis * self.wf[j].normal / ti.abs(dis) * (ti.abs(dis) + delta_n / 2.0)
            normal = self.wf[j].normal
            tangent = vec2(-self.wf[j].normal[0], self.wf[j].normal[1])
            self.wcf[i, j].position = self.particles[i].pos + r_i

            rel_vel = self.particles[i].vel
            v_c = self.particles[i].ang_vel * tm.length(r_i)
            rel_vn = (rel_vel.dot(normal)) * normal
            rel_vt = (rel_vel.dot(tangent) + v_c) * tangent
            e_star = 1.0 / ((1 - self.poissonRatio ** 2) / self.elasticModulus +
                            (1 - self.wall_poissonRatio ** 2) / self.wall_elasticModulus)
            g_star = 0.5 / ((2.0 - self.poissonRatio) * (1.0 + self.poissonRatio) / self.elasticModulus
                            + (2.0 - self.wall_poissonRatio) * (1.0 + self.wall_poissonRatio)
                            / self.wall_elasticModulus)
            r_star = self.particles[i].radius
            m_star = self.particles[i].mass
            beta = (1.0 / tm.sqrt(1.0 + (tm.pi / ti.log(self.wall_restitution_coef)) ** 2))

            kn = 4.0 / 3.0 * e_star * tm.sqrt(r_star * delta_n)
            cn = 2.0 * beta * tm.sqrt(5.0 / 6.0 * kn * m_star)
            ks = 8 * g_star * tm.sqrt(r_star * delta_n)
            cs = 2.0 * beta * tm.sqrt(5.0 / 6.0 * ks * m_star)

            Fn = kn * delta_n * normal - cn * rel_vn

            shear_increment = rel_vt * self.dt
            self.wcf[i, j].shear_displacement += shear_increment

            Fs = ks * tm.dot(self.wcf[i, j].shear_displacement, tangent) * tangent - cs * rel_vt

            Fn_abs = tm.length(Fn)
            Fs_abs = tm.length(Fs)

            if Fs_abs > self.fraction_coef * Fn_abs:
                Fs = (self.fraction_coef * Fn_abs) * Fs / Fs_abs
                self.wcf[i, j].shear_displacement[0] = Fs[0] / ks
                self.wcf[i, j].shear_displacement[1] = Fs[1] / ks

            ti.atomic_add(self.particles[i].force, Fn + Fs)
            # torque1 = r_i.cross(Fs)
            # ti.atomic_add(self.particles[i].torque, torque1)

    @ti.kernel
    def resolve_wall(self):
        n_p, n_w = self.particle_num, self.n_walls
        for i, j in ti.ndrange(n_p, n_w):
            if self.wcf[i, j].isActive:
                dis = ti.abs(tm.dot(self.particles[i].pos, self.wf[j].normal) - self.wf[j].distance)
                if dis >= self.particles[i].radius:
                    self.wcf[i, j].isActive = 0
                    self.wcf[i, j].shear_displacement = vec2(0.0, 0.0)
                    self.wcf[i, j].position = vec2(0.0, 0.0)
                    self.wcf[i, j].normal_force = vec2(0.0, 0.0)
                    self.wcf[i, j].shear_force = vec2(0.0, 0.0)
                else:
                    self.evaluate_wall(i, j)
            else:
                dis = ti.abs(tm.dot(self.particles[i].pos, self.wf[j].normal) - self.wf[j].distance)
                if dis < self.particles[i].radius:
                    self.wcf[i, j] = Contact(isActive=1,
                                             shear_displacement=vec2(0.0, 0.0),
                                             position=vec2(0.0, 0.0),
                                             normal_force=vec2(0.0, 0.0),
                                             shear_force=vec2(0.0, 0.0))
                    self.evaluate_wall(i, j)

    @ti.kernel
    def clear_state(self):
        for i in self.particles:
            self.particles[i].force = vec2(0.0, 0.0)  # apply gravity
            self.particles[i].torque = 0.0

    @ti.kernel
    def apply_body_force(self):
        for i in self.particles:
            self.particles[i].force += vec2(0, self.gravity * self.particles[i].mass)  # apply gravity

    @ti.kernel
    def contact(self):
        self.particles_count.fill(0)

        for i in range(self.particle_num):
            grid_idx = ti.floor(self.particles[i].pos * self.grid_n, int)
            self.particles_count[grid_idx] += 1

        for i in range(self.grid_n):
            sum_tmp = 0
            for j in range(self.grid_n):
                sum_tmp += self.particles_count[i, j]
            self.column_sum[i] = sum_tmp

        self.prefix_sum[0, 0] = 0

        ti.loop_config(serialize=True)
        for i in range(1, self.grid_n):
            self.prefix_sum[i, 0] = self.prefix_sum[i - 1, 0] + self.column_sum[i - 1]

        for i in range(self.grid_n):
            for j in range(self.grid_n):
                if j == 0:
                    self.prefix_sum[i, j] += self.particles_count[i, j]
                else:
                    self.prefix_sum[i, j] = self.prefix_sum[i, j - 1] + self.particles_count[i, j]

                linear_idx = i * self.grid_n + j

                self.list_head[linear_idx] = self.prefix_sum[i, j] - self.particles_count[i, j]
                self.list_cur[linear_idx] = self.list_head[linear_idx]
                self.list_tail[linear_idx] = self.prefix_sum[i, j]

        for i in range(self.particle_num):
            grid_idx = ti.floor(self.particles[i].pos * self.grid_n, int)
            linear_idx = grid_idx[0] * self.grid_n + grid_idx[1]
            grain_location = ti.atomic_add(self.list_cur[linear_idx], 1)
            self.particle_id[grain_location] = i

        if self.particle_num > 100:
            # Fast collision detection
            for i in range(self.particle_num):
                grid_idx = ti.floor(self.particles[i].pos * self.grid_n, int)
                x_begin = ti.max(grid_idx[0] - 1, 0)
                x_end = ti.min(grid_idx[0] + 2, self.grid_n)

                y_begin = ti.max(grid_idx[1] - 1, 0)
                y_end = ti.min(grid_idx[1] + 2, self.grid_n)

                for neigh_i in range(x_begin, x_end):
                    for neigh_j in range(y_begin, y_end):
                        neigh_linear_idx = neigh_i * self.grid_n + neigh_j
                        for p_idx in range(self.list_head[neigh_linear_idx],
                                           self.list_tail[neigh_linear_idx]):
                            j = self.particle_id[p_idx]
                            if i < j:
                                self.resolve(i, j)
        else:
            for i in range(self.particle_num):
                for j in range(i + 1, self.particle_num):
                    self.resolve(i, j)

    @ti.kernel
    def get_energy(self) -> ti.f32:
        energy = 0.0
        for i in self.particles:
            ti.atomic_add(energy, self.particles[i].energy)
        return energy

    @ti.kernel
    def get_average_ang_vel(self) -> ti.f32:
        average_ang_vel = 0.0
        for i in self.particles:
            ti.atomic_add(average_ang_vel, self.particles[i].ang_vel)
        return average_ang_vel

    @ti.kernel
    def get_abs_ang_vel(self) -> ti.f32:
        abs_ang_vel = 0.0
        for i in self.particles:
            ti.atomic_add(abs_ang_vel, ti.abs(self.particles[i].ang_vel))
        return abs_ang_vel

    @ti.kernel
    def apply_fraction(self):
        for i in self.particles:
            force_abs = tm.length(self.particles[i].force)
            # print(f"robot {i} force: {force_abs}")
            if (force_abs <= self.particles[i].mass * self.gravity * self.fraction_coef)\
                    and (tm.length(self.particles[i].vel) < 0.0001):
                self.particles[i].force = vec2(0.0, 0.0)
                self.particles[i].vel = vec2(0.0, 0.0)
            else:
                fraction = vec2(0.0, 0.0)
                if tm.length(self.particles[i].vel) < 0.0001:
                    fraction -= self.particles[i].force / tm.length(self.particles[i].force)
                else:
                    fraction -= self.particles[i].vel / tm.length(self.particles[i].vel)
                fraction *= self.particles[i].mass * self.gravity * self.fraction_coef
                self.particles[i].force += fraction

    def forward(self):
        self.clear_state()
        # self.apply_body_force()
        self.apply_bc()
        self.contact()
        self.apply_fraction()
        if self.n_walls > 0:
            self.resolve_wall()
        self.update()
        self.late_clear_state()

    @ti.kernel
    def change_radius(self, action: ti.types.ndarray()):
        """ Step function for change radius according to action

        Args:
            action: np.ndarray, length = self.particle_num.
                    action =[a_0, a_1, ..., a_n], where a_i in [0, 1].
                    0 - full constraint -> r = self.r_min
                    1 - full expanded -> r = self.r_max.
                    modify self.particles[i].radius
        Returns:
            None.
        """
        for i in self.particles:
            r_i = action[i] * (self.r_max - self.r_min) + self.r_min
            if action[i] < 0.0:
                r_i = self.r_min
            elif action[i] > 1.0:
                r_i = self.r_max
            if r_i > self.particles[i].radius:
                if r_i - self.r_speed > self.particles[i].radius:
                    self.particles[i].radius += self.r_speed
                else:
                    self.particles[i].radius = r_i
            elif r_i < self.particles[i].radius:
                if r_i + self.r_speed < self.particles[i].radius:
                    self.particles[i].radius -= self.r_speed
                else:
                    self.particles[i].radius = r_i

    def step(self, action):
        """ Step function for change radius according to action

        Args:
            action: list or np.ndarray, length = self.particle_num.
                    action =[a_0, a_1, ..., a_n], where a_i in [0, 1].
                    0 - full constraint -> r = self.r_min
                    1 - full expanded -> r = self.r_max.
                    modify self.particles[i].radius
        Returns:
            None.
        """
        if isinstance(action, list):
            action = np.array(action)
        for _ in range(self.action_substeps):
            self.change_radius(action)
            self.forward()

    @ti.kernel
    def wave_policy(self, res_action: ti.types.ndarray(dtype=float, ndim=1), t: float):
        for i in self.particles:
            distance = tm.length(self.particles[i].pos)
            distance /= self.window_size * tm.sqrt(2)
            distance *= 2 * tm.pi
            res_action[i] = tm.sin(distance * 10000 + t*100) * 0.5 + 0.5


def gradient_green(x):
    # 定义浅绿色和深绿色的 RGB 值
    light_green = (0, 255, 0)
    dark_green = (0, 50, 0)

    # 使用线性插值计算对应 x 值的绿色通道值
    green_value = light_green[1] * (1 - x) + dark_green[1] * x
    green_value = green_value.astype(int)
    green_value = green_value * (16 * 16)
    # 返回对应的 RGB 值
    return green_value


if __name__ == "__main__":
    ti.init(arch=ti.gpu, device_memory_GB=6, debug=False)

    parser = argparse.ArgumentParser()
    # ----------------- world settings ----------------------
    parser.add_argument('--window_size', type=int, default=1024,
                        help='size of window')
    parser.add_argument('--particle_num', type=int, default=100,
                        help='number of particles')
    parser.add_argument('--grid_n', type=int, default=64,
                        help='number of particles')
    parser.add_argument('--gravity', type=float, default=9.81,
                        help='gravity of the world')
    parser.add_argument('--dt', type=float, default=0.0001,
                        help='simulation time of a time step')
    parser.add_argument('--action_substeps', type=int, default=100,
                        help='simulation time of a time step')
    parser.add_argument('--substeps', type=int, default=20,
                        help='substep length between visualization')
    parser.add_argument('--contact_num_max', type=int, default=64,
                        help='max contact of individual particle')

    # ------------------ wall settings -----------------
    parser.add_argument('--nWall', type=int, default=0,
                        help='number of walls')
    parser.add_argument('--wall_elasticModulus', type=float, default=1e7,
                        help='elastic modulus of walls')
    parser.add_argument('--wall_poissonRatio', type=float, default=0.45,
                        help='poisson ratio of walls')
    parser.add_argument('--wall_fraction_coef', type=float, default=0.4,
                        help='fraction coefficient of walls')
    parser.add_argument('--wall_restitution_coef', type=float, default=0.01,
                        help='fraction coefficient of walls')

    # ------------------ particle settings --------------------
    parser.add_argument('--radius_min', type=float, default=3.0,
                        help='min radius of particles')
    parser.add_argument('--radius_max', type=float, default=4.0,
                        help='max radius of particles')
    parser.add_argument('--radius_speed', type=float, default=0.01,
                        help='radius changing speed of particles')
    parser.add_argument('--density', type=float, default=0.02,
                        help='density of particles')
    parser.add_argument('--elasticModulus', type=float, default=1e6,
                        help='elastic modulus of particles')
    parser.add_argument('--poissonRatio', type=float, default=0.45,
                        help='poisson ratio of particles')
    parser.add_argument('--fraction_coef', type=float, default=0.4,
                        help='fraction coefficient of particles')
    parser.add_argument('--restitution_coef', type=float, default=0.01,
                        help='fraction coefficient of particles')
    parser.add_argument('--rotation', default=False, action='store_true',
                        help='will simulate rotate particle if true')

    # ------------------ save settings ----------------------
    parser.add_argument('--save_path', type=str, default='illustrations',
                        help='save path of output video')
    parser.add_argument('--save', default=False, action='store_true',
                        help='will save data if true')

    args = parser.parse_args()

    window_size = args.window_size
    n = args.particle_num
    grid_n = args.grid_n
    SAVE_FRAMES = args.save

    particles = ParticleHertzMindlin(args)
    gui = ti.GUI('Taichi DEM', (window_size, window_size))
    step = 0

    if SAVE_FRAMES:
        os.makedirs('output', exist_ok=True)
    time_s = 0
    time_re = 0
    frame_num = 0
    while (not gui.get_event(ti.GUI.ESCAPE)) and (time_re < 1):
        for _ in range(args.substeps):
            time_s += args.dt * args.substeps
            time_re += args.dt * args.substeps
            if time_s > 4 * tm.pi:
                time_s = 0
            action = np.zeros(n)
            particles.wave_policy(action, time_s)
            particles.step(action)

        pos = particles.particles.pos.to_numpy()
        r = particles.particles.radius.to_numpy() * window_size
        r_color = (r.copy() - args.radius_min) / (args.radius_max - args.radius_min)
        r_color.astype('int32')
        colors = gradient_green(r_color)
        gui.clear(0xFFFFFF)
        gui.circles(pos, radius=r, color=colors)
        frame_num += 1
        # gui.circles(pos, radius=r)
        if args.nWall > 0:
            X = np.zeros((4, 2))
            Y = np.zeros((4, 2))
            for i, j, k in [(0, 2, 3), (1, 2, 1), (2, 0, 1), (3, 0, 3)]:
                X[i] = np.array([abs(particles.wf[j].distance), abs(particles.wf[k].distance)])
                Y[i] = X[(i + 1) % 4][::-1]
            gui.lines(begin=X, end=Y, radius=2, color=0xEEEEF0)

        # print("energy: {}".format(particles.get_energy()))
        # print("angle vel: {}".format(particles.get_average_ang_vel()))
        # print("abs vel: {}".format(particles.get_abs_ang_vel()))

        if SAVE_FRAMES:
            gui.show(f'{args.save_path}/{step:06d}.png')
        else:
            gui.show()
        step += 1

    if SAVE_FRAMES:
        fps = int(frame_num / time_re)
        video_commend = f"D:/softwares/ffmpeg/ffmpeg/bin/ffmpeg -y -r {fps}"
        video_commend += f" -i {args.save_path}/%6d.png"
        video_commend += f" -pix_fmt yuv420p -c:v libx264"
        video_commend += f" output.mp4"
        print(video_commend)
        subprocess.getstatusoutput(video_commend)
