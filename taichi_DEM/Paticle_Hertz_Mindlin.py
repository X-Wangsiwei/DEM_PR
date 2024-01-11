import taichi as ti
import taichi.math as tm
import math
import os
import numpy as np
import argparse
import subprocess
##以上为导入模块

##以下为定义变量
vec2 = tm.vec2
vec3 = tm.vec3
vec4 = tm.vec4

##此处tik_system中的两个浮点型变量具体含义是什么？
@ti.dataclass
class tik_system:
    tik: ti.f32
    shake_sate_machine: ti.f32

##此处定义了名为粒子旋转的类，三个段落分明定义了基本属性、平动特性、转动特性。
##其中基本特性包括编号，位置，质量，半径，颜色平动特性包括速度、加速度、力、方向、旋转角速度和角加速度，而转动特性包括惯性、角速度
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

##此处定义的类为contact，包括接触双方的编号，接触是否发生，接触导致的位移，接触点位置，以及接触导致的应力，分为切向和法向。
@ti.dataclass
class Contact:
    i: ti.i32
    j: ti.i32
    isActive: ti.i32  # Contact exists: 1 - exist 0 - not exist
    shear_displacement: vec2  # Shear displacement stored in the contact
    position: vec2  # Position of contact point
    normal_force: vec2
    shear_force: vec2

##此类定义了wall，包括法向量和与原点的距离，以及速度，不过我目前还不清楚这个速度是做什么用的。
##墙体的方程为Ax + By - C = 0，这是日后输入墙体信息的格式吗？
@ti.dataclass
class Wall:
    # Wall equation: Ax + By - C = 0
    normal: vec2
    distance: float
    speed: vec2

##这大概是个初始化函数，定义了例子和环境的各种参数
@ti.data_oriented
class ParticleHertzMindlin:
    def __init__(self, args):
        """ python scope data """
        ##以下为python中直接可用的参数

        ##这个部分定义了系统的一些空间规模参数，包括粒子数量、网格大小、网格数量
        # size data
        self.particle_num = args.particle_num
        self.grid_n = args.grid_n
        self.grid_size = 1.0 / self.grid_n  # Simulation domain of size [0, 1]

        ##这个部分定义了一些环境参数，比如窗口大小，重力加速度，时间步长，其中action_substeps和substeps两个参数的含义我此时并不清楚
        # env setting
        self.window_size = args.window_size
        self.gravity = args.gravity
        self.dt = args.dt
        self.action_substeps = args.action_substeps
        self.substeps = args.substeps

        ## 设置模拟墙的数量。参数来自于args对象，其名称是"nWall"。
        ## 这个部分的功能我可能需要结合后面的代码来理解
        self.n_walls = args.nWall
        ## 如果墙的数量大于0，则创建一个名为"wf"的Wall.field对象，其形状由参数"nWall"决定。
        ## 同时创建一个名为"wcf"的Contact.field对象，其形状为(args.particle_num, args.nWall)。
        if args.nWall > 0:
            self.wf = Wall.field(shape=args.nWall)
            self.wcf = Contact.field(shape=(args.particle_num, args.nWall))
        ## 否则，创建一个名为"wf"的Wall.field对象，其形状为1。
        else:
            self.wf = Wall.field(shape=1)
            self.wcf = Contact.field(shape=(args.particle_num, 1))

        ## 设置墙的泊松比、弹性模量、分数系数和恢复系数的参数值。这些参数来自于args对象。
        self.wall_poissonRatio = args.wall_poissonRatio
        self.wall_elasticModulus = args.wall_elasticModulus
        self.wall_fraction_coef = args.wall_fraction_coef
        self.wall_restitution_coef = args.wall_restitution_coef

        ## 定义了粒子的一些性质，
        # particle property

        ##旋转和接触数量的最大值，密度、弹性模量、泊松比、分数系数和恢复系数的参数值。但分数系数是什么？
        self.rotation = args.rotation
        self.contact_num_max = args.contact_num_max
        self.density = args.density
        self.elasticModulus = args.elasticModulus
        self.poissonRatio = args.poissonRatio
        self.fraction_coef = args.fraction_coef  # typical: 0.75
        self.restitution_coef = args.restitution_coef  # typical: 0.01

        ## 定义了粒子的最小初始半径、最大初始半径和半径速度，最小初始半径和最大初始半径还加上了其和窗口的比。
        self.r_min_orig = args.radius_min
        self.r_max_orig = args.radius_max
        self.r_min = args.radius_min / args.window_size  # typical: 0.002
        self.r_max = args.radius_max / args.window_size  # typical: 0.003
        self.r_speed = args.radius_speed / args.window_size # typical: 0.0001
        # self.mass = 0.1


        ## 以下为taichi中可用的参数
        """ taichi scope data """

        # particles
        ## 粒子旋转的基本参数，是一个数组，其元素是对应粒子旋转的角速度吗？
        self.particles = ParticleRotate.field(shape=(self.particle_num,))

        # grid
        ## 这里的head、cur、tail应该是对网格而言的，其意义不明。目前可以知道这三个玩意儿都是二阶矩阵，共有n*n个元素。
        ## 由于grid是二维的，不难猜测矩阵中的各个元素就对应着网格中的各个点，同时head、cur、tail分别对应着各个点的先前节点、当下节点和后续节点。
        self.list_head = ti.field(dtype=ti.i32, shape=self.grid_n * self.grid_n)
        self.list_cur = ti.field(dtype=ti.i32, shape=self.grid_n * self.grid_n)
        self.list_tail = ti.field(dtype=ti.i32, shape=self.grid_n * self.grid_n)

        ## 这一条也是对网格而言的，但是与上面的几条定义式不同，这几条定义都加上了name=，由此推断上面的部分应该是作为内部变量使用，而下面部分则会在后面的代码中出现。
        ## 以下四条分别是粒子计数，关于某一列的参数，prefix我不清楚，id的意思十分明显，就是粒子编号。
        self.particles_count = ti.field(dtype=ti.i32,
                                        shape=(self.grid_n, self.grid_n),
                                        name="particles_count")
        self.column_sum = ti.field(dtype=ti.i32, shape=self.grid_n, name="column_sum")
        self.prefix_sum = ti.field(dtype=ti.i32, shape=(self.grid_n, self.grid_n), name="prefix_sum")
        self.particle_id = ti.field(dtype=ti.i32, shape=self.particle_num, name="particle_id")

        ##关于接触的参数
        self.cf = Contact.field(shape=self.contact_num_max * self.particle_num)  # neighbors for every particle
        self.cfn = ti.field(ti.i32, shape=self.particle_num)  # neighbor counter for every particle

        # shake settings
        ## 定义了shake设置中的参数，包括时间周期、最大移动距离和加速度。
        self.quarter_period = 5000.0 ## 四分之一周期的时间为5000.0，单位是毫秒吗？
        self.max_movement = 0.18 ## shake设置中的最大移动距离为0.18
        self.a_shake = self.max_movement / (self.quarter_period ** 2) ## self.a_shake为加速度，通过将最大移动距离除以时间段平方来计算
        ## 实际上四分之一振荡周期，刚好是开始加速到最大距离半程的时间，故为1/2x=1/2at^2，即有a_shake=max_movement/quarter_period^2

        # self.shake_sate_machine = 0 这行代码被注释掉了，如果取消注释，则该行表示一个名为shake_sate_machine的实例属性被初始化为0。这一变量在tik_system类中定义，其含义至今未知。

        self.direction = vec2(1.0, 0.0) ## shake此处设置的方向为向右，但是这是什么的方向呢？
        
        # self.tik = 0 这行代码被注释掉了。如果取消注释，则该行表示tik被初始化为0。
        # self.tik_sys = tik_system(tik=0, shake_sate_machine=0) 这行代码被注释掉了，如果取消注释，则该行表示创建一个名为tik_sys的tik_system对象，并将tik和shake_sate_machine设置为0。

        ## 接下来这三个句子都和tik_sys有关，我只知道其形式含义，不知道其具体作用。
        self.tik_sys = tik_system.field(shape=(1,))
        ## self.tik_sys是一个field对象，该对象属于tik_system类，并具有形状为(1,)的shape属性。这里可能用于存储某种动态系统或状态。
        self.tik_sys[0].tik = 0
        ## self.tik_sys[0].tik是设置在tik_sys字段中的第一个元素（索引为0）的属性，并将其值设置为0。这可能是对某种动态系统或状态的初始化或配置。
        self.tik_sys[0].shake_sate_machine = 0
        ## self.tik_sys[0].shake_sate_machine是设置在tik_sys字段中的第一个元素的另一个属性，并将其值设置为0。这可能是对某种动态系统或状态的另一种初始化或配置。

        """ value init """
        self.init_func()

    @ti.kernel
    def init_func(self):
        ## 初始化粒子，包括id、位置、半径、质量、速度和角速度。
        for i in self.particles:
            self.particles[i].id = i + 1 ## 此处为粒子赋予id，从1开始计数。
           
            ## 以下几句可以算是对粒子位置的整体安排
            loc = i * 2 * self.r_min ## 计算粒子在网格中的位置，其值为i*2*r_min，从这句代码来看，粒子应该是一字排开，彼此相接实际上这种模式是不可能的，且看后面如何圆回来。
            region_width = int(self.particle_num ** 0.5) * 2 * self.r_min ## 值得注意的是，此处的r_min是窗口单位，其定义语句为self.r_min = args.radius_min / args.window_size，典型值0.002
            ## 这个语句的目的应该是大致地求出粒子以正方形网格分布时的边长，但是这个边长显然是不准确的，考虑到python中转整数的算法，这个取整的实际效果应该是向下取整，因此实际得到的边长比实际值小一些。
            start = [0.5 - region_width / 2, 0.5 - region_width / 2] ## 由于采取窗口单位，region_width的值至多不大于1，窗口中心为0.5处，故此语句代表粒子分布在整个窗口居中的位置
            
            pos = vec2(loc % region_width + start[0],
                       loc // region_width * 2 * self.r_min + start[1])
            ## 这个语句用于计算粒子的具体位置，生成的pos是一个二维向量，其横向坐标为一字排开长度对区域宽度的余数加上起始位置（现在回顾原本的loc，其作用就很明显了），纵坐标为一字排开长度除以区域宽度，再乘以2*r_min再加上起始位置。

            # region_width = int(self.particle_num ** 0.5) * 2 * self.r_max
            # start = [0.5 - region_width / 2, 0.5 - region_width / 2]
            # pos = vec2(loc % region_width + 1.0 * (self.r_max - self.r_min) * ti.random() + start[0],
            #            loc // region_width * 2 * self.r_max + 1.0 * (self.r_max - self.r_min) * ti.random() + start[1])
            ## 这几行好像和上面那一段重复，效果的不同在于，这里粒子位置的生成是随机的。

            ## 以下几句就是对每个粒子的初始化了
            ## 这段代码是关于一个粒子系统的一部分，其中每个粒子具有位置（pos）、半径（radius）、质量（mass）、方向（orient）、转动惯量（inertia）、速度（vel）和角速度（ang_vel）等属性。
            self.particles[i].pos = pos
            # self.particles[i].radius = ti.random() * (self.r_max - self.r_min) + self.r_min
            self.particles[i].radius = self.r_min
            self.particles[i].mass = self.density * tm.pi * \
                                     (self.particles[i].radius * self.window_size) ** 2 
            ## 此处直接用密度计算粒子的质量，然而radius是用self.r_min，密度是否是根据最小值来计算的？如果不是，这会导致计算出的质量与实际不符。
            
            print(f"robot {i} mass: {self.particles[i].mass}, fraction force: {self.particles[i].mass * 9.81 * self.fraction_coef}")
            
            # self.particles[i].mass = 0.1

            ## 这里的一堆语句是关于粒子的各项属性，除了转动惯量由质量半径计算外，其他均为零
            self.particles[i].orient = 0.0
            self.particles[i].inertia = 0.5 * self.particles[i].mass * (
                    (self.particles[i].radius * self.window_size) ** 2)
            self.particles[i].vel = vec2([0.0, 0.0])
            self.particles[i].ang_vel = 0.0

        ## 接下来是关于墙的初始化，包括位置、法向量和速度。看起来墙的初始化并不是通过给出一般方程来进行
        if args.nWall == 4:
            ## 共有四面墙，则初始化四个面
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
        offset = ti.atomic_add(self.cfn[i], 1) ## 为什么这里要用ti.atomic_add？经过查询貌似是为了防止多线程的冲突，具体是这样吗？
        if offset < self.contact_num_max:
            ret = i * self.contact_num_max + offset
        return ret
    ## 这一段的操作我可以理解，但是作用我不完全理解，估计还带看后文。

    @ti.func
    def search_active_contact_offset(self, i, j):
        ret = -1
        for offset in range(self.cfn[i]):
            tmp = ret = i * self.contact_num_max + offset
            if self.cf[tmp].j == j and self.cf[tmp].isActive == 1:
                ret = tmp
                break
        return ret
    ## 顾名思义，这个函数的作用是搜索与粒子i有接触的粒子j的接触信息，具体操作是遍历所有接触信息，如果找到与粒子i有接触的粒子j，则返回其接触信息。至于接触信息具体是什么，还需要细看
    ## 这两段关于接触的函数我都不是很理解，但是现在感觉它们是很有用的。

    @ti.func
    ##顾名思义，这一段用于去除没有接触的粒子，也就是将所有接触信息中与粒子i没有接触的粒子j设置为无效，具体实现方法没有细看
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

    ## 这一段是关于接触的，作用是去除与粒子i没有接触的粒子j，调用了上面定义的remove_inactive_contact函数
    @ti.kernel
    def late_clear_state(self):
        for i in self.particles:
            self.remove_inactive_contact(i)

    
    @ti.kernel

    ## update部分是关于如何更新粒子位置和速度的，其中包含碰撞检测、摩擦力和重力    
    def update(self):
        # self.shake_wall()
        for i in self.particles:
            acc = self.particles[i].force / self.particles[i].mass ## 这里以力为出发点计算加速度，然后更新速度和位置
            self.particles[i].vel += (self.particles[i].acc + acc) * self.dt / 2.0
            self.particles[i].pos += self.particles[i].vel * self.dt + 0.5 * acc * (self.dt ** 2)
            self.particles[i].acc = acc

            if self.rotation: ## 从这里可以看出，self.rotation是一个bool量，以此判断是否需要更新粒子旋转
                ang_acc = self.particles[i].torque / self.particles[i].inertia
                self.particles[i].ang_vel += (self.particles[i].ang_acc + ang_acc) * self.dt / 2.0
                self.particles[i].orient += self.particles[i].ang_vel * self.dt + \
                                            0.5 * ang_acc * (self.dt ** 2)
                self.particles[i].ang_acc = ang_acc
                self.particles[i].inertia = 0.5 * self.particles[i].mass * \
                                            (self.particles[i].radius * self.window_size) ** 2
            
            ## 下面的部分计算出速率和能量
            self.particles[i].speed = tm.length(self.particles[i].vel)
            self.particles[i].energy = \
                - self.particles[i].pos[1] * self.particles[i].mass * self.gravity + \
                0.5 * self.particles[i].mass * self.particles[i].speed ** 2 + \
                0.5 * self.particles[i].inertia * (self.particles[i].ang_vel ** 2)

    ## 这一段问题很大，我几乎完全不清楚tik_sys的含义，我感觉这是个计时器或者计数器，但是对其应用和操作都不明白
    @ti.func
    def shake_wall(self):
        # print(self.tik_sys[0].tik)
        ti.atomic_add(self.tik_sys[0].tik, 1)
        # if self.tik_sys[0].tik % 2 == 0:
        #     print(self.tik_sys[0].tik)
        if self.tik_sys[0].tik > self.quarter_period:
            self.tik_sys[0].tik = 0
            ti.atomic_add(self.tik_sys[0].shake_sate_machine, 1)
            if self.tik_sys[0].shake_sate_machine >= 4:
                self.tik_sys[0].shake_sate_machine = 0

        for j in self.wf:
            if self.tik_sys[0].shake_sate_machine == 0 or self.tik_sys[0].shake_sate_machine == 3:
                self.wf[j].speed += self.a_shake
                sign = tm.sign(tm.dot(self.direction, self.wf[j].normal))
                self.wf[j].distance -= sign * tm.dot(self.direction * self.wf[j].speed,
                                                     self.wf[j].normal)
            else:
                self.wf[j].speed -= self.a_shake
                sign = tm.sign(tm.dot(self.direction, self.wf[j].normal))
                self.wf[j].distance -= sign * tm.dot(self.direction * self.wf[j].speed,
                                                     self.wf[j].normal)


    @ti.kernel
    def apply_bc(self):
        bounce_coef = 0.3  # Velocity damping  
        ## 定义一个名为bounce_coef的变量，表示速度阻尼系数，用于反弹时的速度衰减

        ## 遍历所有粒子
        for i in self.particles:

            ## 获取粒子坐标
            x = self.particles[i].pos[0]
            y = self.particles[i].pos[1]

            ## 如果粒子接触到y轴的下界
            if y - self.particles[i].radius < 0:  
                ## 将粒子的y坐标设置为粒子半径，将粒子的y速度取反并乘以速度阻尼系数，将粒子的角速度取反并乘以速度阻尼系数
                ## 为什么这里的速度一律取反？若本来粒子在向上运动，但是这里将速度取反后粒子在向下运动，继续接触下界，然后再取反向上，这是不是多此一举？下文中也有这个疑问。
                self.particles[i].pos[1] = self.particles[i].radius
                self.particles[i].vel[1] *= -bounce_coef
                self.particles[i].ang_vel *= bounce_coef

            ## 如果粒子接触到y轴的上界
            elif y + self.particles[i].radius > 1.0:
                ## 将粒子的y坐标设置为1减去粒子半径，将粒子的y速度取反并乘以速度阻尼系数，将粒子的角速度取反并乘以速度阻尼系数
                self.particles[i].pos[1] = 1.0 - self.particles[i].radius
                self.particles[i].vel[1] *= -bounce_coef
                self.particles[i].ang_vel *= bounce_coef

            ## 如果粒子接触到x轴的左界
            if x - self.particles[i].radius < 0:  
                ## 将粒子的x坐标设置为粒子半径，将粒子的x速度取反并乘以速度阻尼系数，将粒子的角速度取反并乘以速度阻尼系数
                self.particles[i].pos[0] = self.particles[i].radius
                self.particles[i].vel[0] *= -bounce_coef
                self.particles[i].ang_vel *= bounce_coef

            ## 如果粒子接触到x轴的右界
            elif x + self.particles[i].radius > 1.0:
                ## 将粒子的x坐标设置为1减去粒子半径，将粒子的x速度取反并乘以速度阻尼系数，将粒子的角速度取反并乘以速度阻尼系数
                self.particles[i].pos[0] = 1.0 - self.particles[i].radius
                self.particles[i].vel[0] *= -bounce_coef
                self.particles[i].ang_vel *= bounce_coef

    @ti.func
    def resolve(self, i, j):
        offset = self.search_active_contact_offset(i, j) ## 调用上面的函数，返回粒子接触信息

        ## 计算相对位置和相对速度
        rel_pos = self.particles[j].pos - self.particles[i].pos
        rel_vel = self.particles[j].vel - self.particles[i].vel
        
        # dist = tm.sqrt(rel_pos[0] ** 2 + rel_pos[1] ** 2) 这行代码被注释掉了，如果没有注释掉，这行代码应该是代表着相对位置的大小。
        
        dist = tm.length(rel_pos) ## 相对位置的大小
        delta = - dist + (self.particles[i].radius + self.particles[j].radius) ## 这行代码是相对位置的大小减去两个粒子半径之和，也就是相对位置与平衡状态的差值
        if delta < 0: ## delta < 0，说明dist大于两个粒子半径之和，粒子不接触
            if offset >= 0: ## 说明至少有一个粒子接触了，然而并没有接触信息，说明这个粒子可能是被其他粒子接触，所以需要将其接触信息删除，各种值都设为零
                self.cf[offset].isActive = 0
                self.cf[offset].shear_displacement = vec2(0.0, 0.0)
                self.cf[offset].position = vec2(0.0, 0.0)
                self.cf[offset].normal_force = vec2(0.0, 0.0)
                self.cf[offset].shear_force = vec2(0.0, 0.0)
            
            # external force
            normal = rel_pos / dist ## 这是相对位置的方向单位向量
            Fn = normal ## 为什么又要赋值给Fn？
            p1, p2 = 0.045 * self.r_min, 0.095 * self.r_min ## 又是一个看不懂的赋值，为什么是0.045和0.095？
            
            ## 接下来这个if语句是计算Fn的，Fn在此处应该是指法向力，在三个不同的距离范围中有三种不一样的算法
            if delta > - p1:
                Fn = 2.5 * normal
            elif delta > - p2:
                Fn = (2.5 + (3.0*0.000015884 / (p2**2) - 2.5) / (p2 - p1) * (-delta - p2)) * normal
            else:
                Fn = 3.0 * 0.000015884 * normal / (delta ** 2)

            ## 这段代码是给i,j粒子添加法向力Fn
            ti.atomic_add(self.particles[i].force, Fn)
            ti.atomic_add(self.particles[j].force, - Fn)

            ## 然而有一个疑问，既然delta < 0，说明两个粒子之间距离大于两个粒子半径之和，那么Fn为什么存在？至少Fn应该是负值，为什么这里Fn可以是正值？

        ## 接下来这一段或许是文档的核心部分，涉及到粒子接触时的算法

        else:  ## 这也就是粒子之间接触的情况，这种情况比不接触的情况要复杂得多，应该会用到Hertz-Mindlin理论
            if offset < 0: ## 看到这里越发发现自己对offset的理解不够，240-260行间的那两段可能需要问问师兄
                offset = self.append_contact_offset(i)
                if offset < 0:
                    print(f"ERROR: coordinate number > "
                          f"set_max_coordinate_number({self.contact_num_max})")
                self.cf[offset] = Contact(i=i, j=j, shear_displacement=vec2(0.0, 0.0))

            # get normal and tangent direction
            ##如其所言，这里计算的是法向和切向方向，法向是直接以相对位置除以距离，切向是法向的垂直向量（逆时针旋转90°）
            normal = rel_pos / dist
            tangent = vec2(-normal[1], normal[0])

            # get contact point and distance to core
            ## 计算接触点坐标，以及其到两个核心的距离
            c_pos = self.particles[i].pos + normal * (self.particles[i].radius - 0.5 * delta) ## 计算接触点坐标
            self.cf[offset].position = c_pos ## 赋值给self.cf[offset].position
            cr_i = c_pos - self.particles[i].pos ## 计算接触点坐标与粒子i的相对位置
            cr_j = c_pos - self.particles[j].pos ## 计算接触点坐标与粒子j的相对位置

            ## 这里计算的大概是粒子表面的线速度，算法是角速度乘以接触点坐标与粒子i的相对位置，也就是角速度乘以半径
            v_c_i = self.particles[i].ang_vel * tm.length(cr_i)
            v_c_j = self.particles[j].ang_vel * tm.length(cr_j)

            ## 计算相对的切向和法向速度
            rel_vn = (rel_vel.dot(normal)) * normal
            rel_vt = rel_vel - rel_vn

            ## 这里要给出相对旋转速度
            if self.rotation:
                rel_vt = (rel_vel.dot(tangent) - v_c_i - v_c_j) * tangent
                ## 相对速度乘以切向方向的单位向量，得到相对切向速度的大小，然后再减去两个角速度，再乘以切向方向单位向量
            else:
                rel_vt = rel_vel - rel_vn
                ## 若没有相对旋转，则直接用相对速度减去相对法向速度，就能得到相对切向速度

            ## 接下来是计算接触力，接触力的算法是Hertz-Mindlin理论
            beta = (1.0 / tm.sqrt(1.0 + (tm.pi / ti.log(self.restitution_coef)) ** 2))
            
            e_star = 2.0 * self.elasticModulus / (1.0 - self.poissonRatio ** 2)
            g_star = self.elasticModulus / ((2.0 - self.poissonRatio) * (1.0 + self.poissonRatio))
            r_star = 1.0 / (1.0 / self.particles[i].radius + 1.0 / self.particles[j].radius)
            m_star = 1.0 / (1.0 / self.particles[i].mass + 1.0 / self.particles[j].mass)

            kn = 4.0 / 3.0 * e_star * tm.sqrt(r_star * delta)
            cn = 2.0 * beta * tm.sqrt(5.0 / 6.0 * kn * m_star)
            Fn = kn * delta * normal - cn * rel_vn

            ks = 8 * g_star * tm.sqrt(r_star * delta)
            cs = 2.0 * beta * tm.sqrt(5.0 / 6.0 * ks * m_star)
            shear_increment = rel_vt * self.dt
            self.cf[offset].shear_displacement += shear_increment
            Fs_try = ks * tm.dot(self.cf[offset].shear_displacement, tangent) * tangent
            Fs = Fs_try - cs * rel_vt

            Fn_abs = tm.length(Fn)
            Fs_abs = tm.length(Fs)

            ## 这一段我已经在Obsidian中打出公式，一些部分和Hertz-Mindlin的公式重合，另一部分可能需要讲解
            # print("Fs: {}, Fn: {}, Fs_abs: {}, Fn_abs: {}, kn: {}, ks: {}".format(Fs, Fn, Fs_abs, Fn_abs,
            #                                                                       kn, ks))

            ## 切向力大于最大静摩擦力，也就是产生了相对滑动
            if Fs_abs > self.fraction_coef * Fn_abs:
                Fs = (self.fraction_coef * Fn_abs) * Fs / Fs_abs ## 在此情况下，最大静摩擦力赋值给切向力
                self.cf[offset].shear_displacement[0] = Fs[0] / ks ## 矢量情况下的赋值
                self.cf[offset].shear_displacement[1] = Fs[1] / ks

            ti.atomic_add(self.particles[i].force, -(Fn + Fs)) ## 将切向力和法向力叠加到合外力中
            ti.atomic_add(self.particles[j].force, Fn + Fs)
            torque1 = cr_i.cross(-Fs) ## 转矩和切向力的叉乘
            torque2 = cr_j.cross(Fs)
            ti.atomic_add(self.particles[i].torque, torque1) ## 将转矩叠加到总力矩中
            ti.atomic_add(self.particles[j].torque, torque2)

            self.cf[offset].normal_force = Fn ## 将法向力和切向力赋值
            self.cf[offset].shear_force = Fs

    # @ti.func 装饰器，标记函数为TensorRT插件函数
    @ti.func
    def evaluate_wall(self, i, j):
        ## 计算粒子i与壁面j的相对位置
        dis = tm.dot(self.particles[i].pos, self.wf[j].normal) - self.wf[j].distance
        ## 计算粒子间的间隙
        gap = dis - self.particles[i].radius
        
        ## 如果间隙大小为负，说明粒子与壁面接触
        if gap < 0:
            ## 若粒子与壁面接触，则粒子表面速度乘以壁面法向量，得到粒子与壁面的相对速度
            self.particles[i].vel += self.wf[j].speed * self.direction
            ## 计算粒子与壁面的相对距离大小
            delta_n = ti.abs(gap)
            # 根据相对距离和壁面法向量计算碰撞点在粒子上的投影点
            r_i = - dis * self.wf[j].normal / ti.abs(dis) * (ti.abs(dis) + delta_n / 2.0)
            # 计算碰撞点的法向量和切线向量
            normal = self.wf[j].normal
            tangent = vec2(-self.wf[j].normal[0], self.wf[j].normal[1])
            # 根据粒子位置和碰撞点投影计算碰撞点的位置
            self.wcf[i, j].position = self.particles[i].pos + r_i

            # 计算粒子相对速度在法线和切线方向的分量
            rel_vel = self.particles[i].vel
            v_c = self.particles[i].ang_vel * tm.length(r_i)
            rel_vn = (rel_vel.dot(normal)) * normal
            rel_vt = (rel_vel.dot(tangent) + v_c) * tangent
            # 计算弹性系数和泊松比等参数
            e_star = 1.0 / ((1 - self.poissonRatio ** 2) / self.elasticModulus + (1 - self.wall_poissonRatio ** 2) / self.wall_elasticModulus)
            g_star = 0.5 / ((2.0 - self.poissonRatio) * (1.0 + self.poissonRatio) / self.elasticModulus + (2.0 - self.wall_poissonRatio) * (1.0 + self.wall_poissonRatio) / self.wall_elasticModulus)
            r_star = self.particles[i].radius
            m_star = self.particles[i].mass
            beta = (1.0 / tm.sqrt(1.0 + (tm.pi / ti.log(self.wall_restitution_coef)) ** 2))

            # 根据弹性系数等参数计算法向力和切向力的系数
            kn = 4.0 / 3.0 * e_star * tm.sqrt(r_star * delta_n)
            cn = 2.0 * beta * tm.sqrt(5.0 / 6.0 * kn * m_star)
            ks = 8 * g_star * tm.sqrt(r_star * delta_n)
            cs = 2.0 * beta * tm.sqrt(5.0 / 6.0 * ks * m_star)

            ## 根据法向力和切向力的系数以及各个方向的相对速度计算碰撞力
            Fn = kn * delta_n * normal - cn * rel_vn

            ## 计算切向位移的变化量，并累加到总切向位移上
            shear_increment = rel_vt * self.dt
            self.wcf[i, j].shear_displacement += shear_increment

            ## 计算切向力
            Fs = ks * tm.dot(self.wcf[i, j].shear_displacement, tangent) * tangent - cs * rel_vt

            ## 计算切向力和法向力的大小
            Fn_abs = tm.length(Fn)
            Fs_abs = tm.length(Fs)

            ## 如果切向力超过最大静摩擦力，则调整切向力
            if Fs_abs > self.fraction_coef * Fn_abs:
                Fs = (self.fraction_coef * Fn_abs) * Fs / Fs_abs
                self.wcf[i, j].shear_displacement[0] = Fs[0] / ks
                self.wcf[i, j].shear_displacement[1] = Fs[1] / ks

            ## 计算接触力的力矩，并叠加到粒子的总力矩中
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
                    fraction = - self.particles[i].force / tm.length(self.particles[i].force)
                else:
                    fraction = - self.particles[i].vel / tm.length(self.particles[i].vel)
                fraction *= self.particles[i].mass * self.gravity  * self.fraction_coef
                self.particles[i].force += fraction / self.window_size

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


if __name__ == "__main__":
    ti.init(arch=ti.gpu, device_memory_GB=6, debug=False)

    parser = argparse.ArgumentParser()
    # ----------------- world settings ----------------------
    parser.add_argument('--window_size', type=int, default=1024,
                        help='size of window')
    parser.add_argument('--particle_num', type=int, default=36,
                        help='number of particles')
    parser.add_argument('--grid_n', type=int, default=64,
                        help='number of particles')
    parser.add_argument('--gravity', type=float, default=9.81,
                        help='gravity of the world')
    parser.add_argument('--dt', type=float, default=0.0001,
                        help='simulation time of a time step')
    parser.add_argument('--action_substeps', type=int, default=200,
                        help='simulation time of a time step')
    parser.add_argument('--substeps', type=int, default=2,
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
    parser.add_argument('--wall_fraction_coef', type=float, default=0.75,
                        help='fraction coefficient of walls')
    parser.add_argument('--wall_restitution_coef', type=float, default=0.01,
                        help='fraction coefficient of walls')

    # ------------------ particle settings --------------------
    parser.add_argument('--radius_min', type=float, default=4.0,
                        help='min radius of particles')
    parser.add_argument('--radius_max', type=float, default=6.0,
                        help='max radius of particles')
    parser.add_argument('--radius_speed', type=float, default=0.01,
                        help='radius changing speed of particles')
    parser.add_argument('--density', type=float, default=0.5,
                        help='density of particles')
    parser.add_argument('--elasticModulus', type=float, default=1e7,
                        help='elastic modulus of particles')
    parser.add_argument('--poissonRatio', type=float, default=0.49,
                        help='poisson ratio of particles')
    parser.add_argument('--fraction_coef', type=float, default=0.75,
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
    while not gui.get_event(ti.GUI.ESCAPE):
        for _ in range(particles.substeps):
            time_s += args.dt * args.action_substeps
            if time_s > 4*tm.pi:
                time_s = 0
            # action = np.random.random(n)
            action = np.zeros(n)
            particles.wave_policy(action, time_s)
            # print(action[0:10])
            particles.step(action)

        pos = particles.particles.pos.to_numpy()
        r = particles.particles.radius.to_numpy() * window_size
        # print(f"max r: {np.max(r)}, min r: {np.min(r)}")
        r_color = (r.copy() - args.radius_min) / (args.radius_max - args.radius_min)
        r_color *= 255
        r_color.astype('int32')
        indices = np.zeros(particles.particle_num, dtype=int)
        w = particles.particles.ang_vel.to_numpy()

        colors = 255*(16**4) + r_color*(16**2) + 255
        colors = colors
        # gui.circles(pos, radius=r, color=colors)
        gui.circles(pos, radius=r)
        if args.nWall > 0:
            X1 = abs(particles.wf[0].distance)
            X2 = abs(particles.wf[1].distance)
            Y1 = abs(particles.wf[2].distance)
            Y2 = abs(particles.wf[3].distance)
            # print(X1, X2, Y1, Y2)

            X = np.zeros((4, 2))
            X[0] = np.array([X1, Y1])
            X[1] = np.array([X1, Y2])
            X[2] = np.array([X2, Y2])
            X[3] = np.array([X2, Y1])
            Y = np.zeros((4, 2))
            Y[0] = np.array([X1, Y2])
            Y[1] = np.array([X2, Y2])
            Y[2] = np.array([X2, Y1])
            Y[3] = np.array([X1, Y1])
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
        fps = int(1.0 / (args.dt * args.substeps * args.action_substeps))
        video_commend = f"D:/softwares/ffmpeg/ffmpeg/bin/ffmpeg -y -r {fps}"
        video_commend += f" -i {args.save_path}/%6d.png"
        video_commend += f" -pix_fmt yuv420p -c:v libx264"
        video_commend += f" output.mp4"
        print(video_commend)
        subprocess.getstatusoutput(video_commend)
