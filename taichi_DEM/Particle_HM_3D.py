from math import pi
import taichi as ti
import taichi.math as tm
import os
import numpy as np
import time

ti.init(ti.gpu, default_fp=64, device_memory_GB=6, debug=False)

# =====================================
# Type Definition
# =====================================
Real = ti.f64
Integer = ti.i32
# Byte = ti.i8
Vector2 = ti.types.vector(2, Real)
Vector3 = ti.types.vector(3, Real)
Vector4 = ti.types.vector(4, Real)
Vector3i = ti.types.vector(3, Integer)
Vector2i = ti.types.vector(2, Integer)
Matrix3x3 = ti.types.matrix(3, 3, Real)

DEMMatrix = Matrix3x3

