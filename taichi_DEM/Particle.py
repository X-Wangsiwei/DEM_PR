import taichi as ti
import math
import os
import numpy as np

vec = ti.math.vec2


@ti.dataclass
class Particle:
    id: ti.i32  # id of particle
    p: vec     # position
    m: ti.f32  # mass
    r: ti.f32  # radius
    s: ti.f32  # speed value
    v: vec     # velocity
    a: vec     # acceleration
    f: vec     # external force
    c: ti.u32  # color
    o: ti.f32  # orientation
    I: ti.f32  # inertia
    w: ti.f32  # angular velocity
    L: ti.f32  # angular acceleration
    T: ti.f32  # torque


@ti.data_oriented
class ParticleWorld:
    def __init__(self, n, grid_n):
        # python scope data
        self.density = 100.0
        self.stiffness = 8e3
        self.restitution_coef = 0.01
        # self.shear = 0
        self.gravity = -9.81
        self.dt = 0.0001  # Larger dt might lead to unstable results.
        self.substeps = 100
        self.rotation = True
        self.friction_coef = 0.75  # 0.75
        self.shear_fraction = 0.04
        self.particle_num = n
        self.grid_n = grid_n
        self.contact_num_max = 10

        # taichi scope data
        self.old_ct = ti.field(dtype=ti.i32, shape=(n, self.contact_num_max))  # contact in the prev
        self.cur_ct = ti.field(dtype=ti.i32, shape=(n, self.contact_num_max))  # contact in the curr
        self.old_fs = ti.field(dtype=ti.f32, shape=(n, self.contact_num_max))  # shear force in the prev
        self.cur_fs = ti.field(dtype=ti.f32, shape=(n, self.contact_num_max))  # shear force in the curr

        self.gf = Particle.field(shape=(n,))  # particle info

        self.list_head = ti.field(dtype=ti.i32, shape=grid_n * grid_n)
        self.list_cur = ti.field(dtype=ti.i32, shape=grid_n * grid_n)
        self.list_tail = ti.field(dtype=ti.i32, shape=grid_n * grid_n)

        self.grain_count = ti.field(dtype=ti.i32,
                                    shape=(grid_n, grid_n),
                                    name="grain_count")
        self.column_sum = ti.field(dtype=ti.i32, shape=grid_n, name="column_sum")
        self.prefix_sum = ti.field(dtype=ti.i32, shape=(grid_n, grid_n), name="prefix_sum")
        self.particle_id = ti.field(dtype=ti.i32, shape=n, name="particle_id")


        self.grid_size = 1.0 / self.grid_n  # Simulation domain of size [0, 1]
        print(f"Grid size: {self.grid_n}x{self.grid_n}")

        # grain_r_min = 0.0775
        # grain_r_max = 0.1175

        self.grain_r_min = 0.002
        self.grain_r_max = 0.003

        self.init_func()

    @ti.kernel
    def init_func(self):
        for i in self.gf:
            self.gf[i].id = i + 1
            # Spread grains in a restricted area.
            loc = i * self.grid_size
            padding = 0.1
            region_width = 0.6 # 1.0 - padding * 2
            pos = vec(loc % region_width + self.grid_size * ti.random() * 0.2 + 0.05,
                      loc // region_width * self.grid_size + self.grid_size * 0.7)
            self.gf[i].p = pos
            self.gf[i].r = ti.random() * (self.grain_r_max - self.grain_r_min) + self.grain_r_min
            # self.gf[i].m = self.density * math.pi * self.gf[i].r**2
            self.gf[i].m = 0.01
            self.gf[i].o = 0.0
            # 1/2*m*r^2 is interia for solid disk/cylinder along z axis
            self.gf[i].I = 0.5 * self.gf[i].m * (self.gf[i].r ** 2)
            self.gf[i].s = ti.sqrt(self.gf[i].v[0]**2 + self.gf[i].v[1]**2)

    @ti.kernel
    def update(self):
        for i in self.gf:
            # apply Euler
            a = self.gf[i].f / self.gf[i].m
            self.gf[i].v += (self.gf[i].a + a) * self.dt / 2.0
            self.gf[i].p += self.gf[i].v * self.dt + 0.5 * a * (self.dt ** 2)
            self.gf[i].a = a

            # cal rotation
            L = self.gf[i].T / self.gf[i].I
            self.gf[i].w += (self.gf[i].L + L) * self.dt / 2.0
            self.gf[i].o += self.gf[i].w * self.dt + 0.5 * L * (self.dt**2)
            self.gf[i].L = L

            for k in range(self.contact_num_max):
                self.old_ct[i, k] = self.cur_ct[i, k]
                self.cur_ct[i, k] = 0
                self.old_fs[i, k] = self.cur_fs[i, k]

            self.gf[i].s = ti.sqrt(self.gf[i].v[0]**2 + self.gf[i].v[1]**2)

    @ti.kernel
    def apply_bc(self):
        bounce_coef = 0.3  # Velocity damping
        for i in self.gf:
            x = self.gf[i].p[0]
            y = self.gf[i].p[1]

            if y - self.gf[i].r < 0:
                self.gf[i].p[1] = self.gf[i].r
                self.gf[i].v[1] *= -bounce_coef
                self.gf[i].w *= bounce_coef

            elif y + self.gf[i].r > 1.0:
                self.gf[i].p[1] = 1.0 - self.gf[i].r
                self.gf[i].v[1] *= -bounce_coef
                self.gf[i].w *= bounce_coef

            if x - self.gf[i].r < 0:
                self.gf[i].p[0] = self.gf[i].r
                self.gf[i].v[0] *= -bounce_coef
                self.gf[i].w *= bounce_coef

            elif x + self.gf[i].r > 1.0:
                self.gf[i].p[0] = 1.0 - self.gf[i].r
                self.gf[i].v[0] *= -bounce_coef
                self.gf[i].w *= bounce_coef

    @ti.func
    def resolve(self, i, j):
        # find (i, j) exist or not in old contact
        old_fs_tmp = self.gf[j].m - self.gf[j].m
        for k in range(self.contact_num_max):
            if self.old_ct[i, k] == self.gf[j].id:
                old_fs_tmp = self.old_fs[i, k]

        rel_pos = self.gf[j].p - self.gf[i].p
        rel_vel = self.gf[j].v - self.gf[i].v
        dist = ti.sqrt(rel_pos[0] ** 2 + rel_pos[1] ** 2)
        delta = -dist + self.gf[i].r + self.gf[j].r  # delta = d - (r_i + r_j)
        if delta > 0:  # in contact
            # update contact
            tmp_index = 0
            for k in range(self.contact_num_max):
                if self.cur_ct[i, k] == 0:
                    pass
                else:
                    tmp_index += 1
            self.cur_ct[i, tmp_index] = self.gf[j].id
            normal = rel_pos / dist
            fn = delta * self.stiffness
            Fn = fn * normal

            # shear force
            tangent = vec(-normal[1], normal[0])

            c_pos = self.gf[i].p + normal*(self.gf[i].r - 0.5 * delta)
            cr1 = c_pos - self.gf[i].p
            cr2 = c_pos - self.gf[j].p
            """
            dist_cr1 = ti.sqrt(cr1[0]**2 + cr1[1]**2)
            dist_cr2 = ti.sqrt(cr2[0]**2 + cr2[1]**2)
            """
            # rel_vn = (rel_vel.dot(normal)) * normal
            # rel_vt = rel_vel - rel_vn
            rel_ds = (rel_vel.dot(tangent)) * self.dt
            del_fs = -self.shear_fraction * self.stiffness * rel_ds
            fs = (old_fs_tmp + del_fs) / 2
            # fs = del_fs

            # friction check
            max_fs = self.friction_coef * fn
            if fs > max_fs:
                self.cur_fs[i, tmp_index] = fs
                fs = max_fs
            elif fs < -max_fs:
                self.cur_fs[i, tmp_index] = fs
                fs = -max_fs
            Fs = fs * tangent
            # Damping force
            M = (self.gf[i].m * self.gf[j].m) / (self.gf[i].m + self.gf[j].m)
            K = self.stiffness
            beta = (1. / ti.sqrt(1. + (math.pi / ti.log(self.restitution_coef)) ** 2))
            C = 2. * beta * ti.sqrt(K * M)
            Vn = rel_vel.dot(normal)
            Vs = rel_vel.dot(tangent)
            if self.rotation:
                Fd = C * Vn * normal + self.shear_fraction * C * Vs * tangent
                self.gf[i].f += Fd - Fn - Fs
                self.gf[j].f -= Fd - Fn - Fs
            else:
                Fd = C * Vn * normal
                # Fd = C * rel_vn
                self.gf[i].f += Fd - Fn
                self.gf[j].f -= Fd - Fn
            if self.rotation:
                self.gf[i].T += cr1.cross(Fs)
                self.gf[j].T -= cr2.cross(Fs)
        # elif delta > - 0.0009:
        #     normal = rel_pos / dist
        #     Fn = 2.5 * normal
        #     self.gf[i].f += Fn
        #     self.gf[j].f -= Fn
        # elif delta > - 0.0019:
        #     normal = rel_pos / dist
        #     Fn = (2.5 + (3.0*0.000015884 / (0.0019**2) - 2.5) / 0.001 * (-delta - 0.0009)) * normal
        #     self.gf[i].f += Fn
        #     self.gf[j].f -= Fn
        # else:
        #     normal = rel_pos / dist
        #     Fn = 3.0*0.000015884 * normal / (delta ** 2)
        #     self.gf[i].f += Fn
        #     self.gf[j].f -= Fn


    @ti.kernel
    def contact(self):
        for i in self.gf:
            self.gf[i].f = vec(0, self.gravity * self.gf[i].m)  # apply gravity

        self.grain_count.fill(0)

        for i in range(self.particle_num):
            grid_idx = ti.floor(self.gf[i].p * self.grid_n, int)
            self.grain_count[grid_idx] += 1

        for i in range(self.grid_n):
            sum_tmp = 0
            for j in range(self.grid_n):
                sum_tmp += self.grain_count[i, j]
            self.column_sum[i] = sum_tmp

        self.prefix_sum[0, 0] = 0

        ti.loop_config(serialize=True)
        for i in range(1, self.grid_n):
            self.prefix_sum[i, 0] = self.prefix_sum[i - 1, 0] + self.column_sum[i - 1]

        for i in range(self.grid_n):
            for j in range(self.grid_n):
                if j == 0:
                    self.prefix_sum[i, j] += self.grain_count[i, j]
                else:
                    self.prefix_sum[i, j] = self.prefix_sum[i, j - 1] + self.grain_count[i, j]

                linear_idx = i * self.grid_n + j

                self.list_head[linear_idx] = self.prefix_sum[i, j] - self.grain_count[i, j]
                self.list_cur[linear_idx] = self.list_head[linear_idx]
                self.list_tail[linear_idx] = self.prefix_sum[i, j]

        for i in range(self.particle_num):
            grid_idx = ti.floor(self.gf[i].p * self.grid_n, int)
            linear_idx = grid_idx[0] * self.grid_n + grid_idx[1]
            grain_location = ti.atomic_add(self.list_cur[linear_idx], 1)
            self.particle_id[grain_location] = i

        if self.particle_num > 100:
            # Fast collision detection
            for i in range(self.particle_num):
                grid_idx = ti.floor(self.gf[i].p * self.grid_n, int)
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

    def forward(self):
        self.update()
        self.apply_bc()
        self.contact()


if __name__ == "__main__":
    ti.init(arch=ti.gpu)
    window_size = 1024
    n = 8192
    grid_n = 128
    SAVE_FRAMES = False

    particles = ParticleWorld(n=n, grid_n=grid_n)
    gui = ti.GUI('Taichi DEM', (window_size, window_size))
    step = 0

    # window = ti.ui.Window('Taichi DEM', res=(window_size, window_size), pos=(150, 150))
    # canvas = window.get_canvas()

    if SAVE_FRAMES:
        os.makedirs('output', exist_ok=True)

    while not gui.get_event(ti.GUI.ESCAPE):
        for s in range(particles.substeps):
            particles.forward()


        pos = particles.gf.p.to_numpy()
        r = particles.gf.r.to_numpy() * window_size
        w = particles.gf.s.to_numpy()
        if np.min(w) != np.max(w):
            w = (w - np.min(w)) / (np.max(w) - np.min(w)) * 255
        # w = 254 - w
        gui.circles(pos, radius=r)
        # window.show()

        # canvas.circles(pos, r)
        if SAVE_FRAMES:
            gui.show(f'output/{step:06d}.png')
        else:
            gui.show()
        step += 1
