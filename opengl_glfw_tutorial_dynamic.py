import math
import threading
from time import sleep

import glfw
import glm
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram


class Param(object):
    width = 640
    height = 480
    eye = glm.vec3(0, 0, 3)  # 眼睛的位置（默认z轴的正方向）
    target = glm.vec3(0, 0, 0)  # 瞄准方向的参考点（默认在坐标原点）
    up = glm.vec3(0, 1, 0)  # 定义对观察者而言的上方（默认y轴的正方向）
    near = 0.1
    far = 100
    radians = glm.radians(45)
    projection = glm.perspective(radians, width / height, near, far)
    view = glm.lookAt(eye, target, up)
    pan_start = None
    orbit_start = None
    pivot_world = None
    add = True


vertex_src = """
# version 330
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;
uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;
out vec3 v_color;
void main()
{
    gl_Position = projection * view * model * vec4(a_position, 1.0);
    v_color = a_color;
}
"""  # vec4(a_position, 1); 1指物体大小，越大越小

fragment_src = """
# version 330
in vec3 v_color;
out vec4 out_color;
void main()
{
    out_color = vec4(v_color, 1.0);   
}
"""
if not glfw.init():
    print("Cannot initialize GLFW")
    exit()
window = glfw.create_window(Param.width, Param.height, "Hello Triangle", None, None)
if not window:
    glfw.terminate()


def depth(x, y):
    depth_buffer = glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
    depth_ = float(depth_buffer[0][0])
    if depth_ == 1:
        pt_drag = glm.vec3(0, 0, 0)
        clip_pos = Param.projection * Param.view * glm.vec4(pt_drag, 1)
        ndc_pos = glm.vec3(clip_pos) / clip_pos.w
        if -1 < ndc_pos.z < 1:
            depth_ = ndc_pos.z * 0.5 + 0.5
    return depth_


def window_size_callback(w, width, height):
    glViewport(0, 0, width, height)
    Param.width, Param.height = width, height
    Param.projection = glm.perspective(Param.radians, width / height, Param.near, Param.far)


def mouse_button_callback(w, button, action, mods):
    x_pos, y_pos = glfw.get_cursor_pos(w)
    y_pos = Param.height - y_pos
    if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
        Param.pan_start = glm.vec3(x_pos, y_pos, depth(x_pos, y_pos))
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        Param.orbit_start = glm.vec3(x_pos, y_pos, depth(x_pos, y_pos))
        Param.pivot_world = glm.vec3(0, 0, 0)


def cursor_pos_callback(w, x_pos, y_pos):
    y_pos = Param.height - y_pos
    # get view matrix and  viewport rectangle
    view, inv_view = Param.view, glm.inverse(Param.view)
    view_rect = glm.vec4(0, 0, Param.width, Param.height)
    if glfw.get_mouse_button(w, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
        # get drag start and end
        wnd_from = Param.pan_start
        wnd_to = glm.vec3(x_pos, y_pos, Param.pan_start[2])
        Param.pan_start = wnd_to

        # get projection and window matrix
        inv_proj = glm.inverse(Param.projection)
        inv_wnd = glm.translate(glm.mat4(1), glm.vec3(-1, -1, -1))
        inv_wnd = glm.scale(inv_wnd, glm.vec3(2 / view_rect[2], 2 / view_rect[3], 2))
        inv_wnd = glm.translate(inv_wnd, glm.vec3(view_rect[0], view_rect[1], 0))

        # calculate drag start and world coordinates
        pt_h_world = [inv_view * inv_proj * inv_wnd * glm.vec4(*pt, 1) for pt in [wnd_from, wnd_to]]
        pt_world = [glm.vec3(pt_h) / pt_h.w for pt_h in pt_h_world]

        # calculate drag world translation
        world_vec = pt_world[1] - pt_world[0]

        # translate view position and update view matrix
        inv_view = glm.translate(glm.mat4(1), world_vec * -1) * inv_view
        view = glm.inverse(inv_view)
    elif glfw.get_mouse_button(w, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
        # get the drag start and end
        wnd_from = Param.orbit_start
        wnd_to = glm.vec3(x_pos, y_pos, Param.orbit_start[2])
        Param.orbit_start = wnd_to

        # calculate the pivot, rotation axis and angle
        pivot_view = glm.vec3(view * glm.vec4(*Param.pivot_world, 1))
        orbit_dir = wnd_to - wnd_from

        # get the projection of the up vector to the view port
        # TODO

        # calculate the rotation components for the rotation around the view space x axis and the world up vector
        orbit_dir_x = glm.vec2(0, 1)
        orbit_vec_x = glm.vec2(0, orbit_dir.y)
        orbit_dir_up = glm.vec2(1, 0)
        orbit_vec_up = glm.vec2(orbit_dir.x, 0)

        # calculate the rotation matrix around the view space x axis through the pivot
        rot_pivot_x = glm.mat4(1)
        if glm.length(orbit_vec_x) > 0.5:
            axis_x = glm.vec3(-1, 0, 0)
            angle_x = glm.dot(orbit_dir_x, glm.vec2(orbit_vec_x.x / (view_rect[2] - view_rect[0]),
                                                    orbit_vec_x.y / (view_rect[3] - view_rect[1]))) * math.pi
            rot_mat_x = glm.rotate(glm.mat4(1), angle_x, axis_x)
            rot_pivot_x = glm.translate(glm.mat4(1), pivot_view) * rot_mat_x * glm.translate(glm.mat4(1), -pivot_view)
        # calculate the rotation matrix around the world space up vector through the pivot
        rot_pivot_up = glm.mat4(1)
        if glm.length(orbit_vec_up) > 0.5:
            axis_up = glm.vec3(0, 1, 0)
            angle_up = glm.dot(orbit_dir_up, glm.vec2(
                orbit_vec_up.x / (view_rect[2] - view_rect[0]),
                orbit_vec_up.y / (view_rect[3] - view_rect[1]))) * math.pi
            rot_mat_up = glm.rotate(glm.mat4(1), angle_up, axis_up)
            rot_pivot_up = glm.translate(
                glm.mat4(1), Param.pivot_world) * rot_mat_up * glm.translate(glm.mat4(1), -Param.pivot_world)
        # transform and update view matrix
        view = rot_pivot_x * view * rot_pivot_up

    Param.view = view


def scroll_callback(w, x_offset, y_offset):
    x, y = glfw.get_cursor_pos(window)
    y = Param.height - y
    view_rect = glm.vec4(0, 0, Param.width, Param.height)

    # get view, projection and window matrix
    proj, inv_proj = Param.projection, glm.inverse(Param.projection)
    view, inv_view = Param.view, glm.inverse(Param.view)
    inv_wnd = glm.translate(glm.mat4(1), glm.vec3(-1, -1, -1))
    inv_wnd = glm.scale(inv_wnd, glm.vec3(2 / view_rect[2], 2 / view_rect[3], 2))
    inv_wnd = glm.translate(inv_wnd, glm.vec3(view_rect[0], view_rect[1], 0))
    wnd = glm.inverse(inv_wnd)

    # get world space position on view ray
    pt_wnd = glm.vec3(x, y, 1.0)
    # pt_world  = glm.unProject(pt_wnd, view, proj, vp_rect)
    pt_h_world = inv_view * inv_proj * inv_wnd * glm.vec4(*pt_wnd, 1)
    pt_world = glm.vec3(pt_h_world) / pt_h_world.w

    # get view position
    eye = glm.vec3(inv_view[3])

    # get "zoom" direction and amount
    ray_cursor = glm.normalize(pt_world - eye)

    # translate view position and update view matrix
    inv_view = glm.translate(glm.mat4(1), ray_cursor * y_offset) * inv_view

    # return new view matrix
    Param.view = glm.inverse(inv_view)


glfw.make_context_current(window)
glfw.set_window_size_callback(window, window_size_callback)
glfw.set_mouse_button_callback(window, mouse_button_callback)
glfw.set_cursor_pos_callback(window, cursor_pos_callback)
glfw.set_scroll_callback(window, scroll_callback)

# (x,y,z,r,g,b) red 1.0, 0.0, 0.0,  green 0.0, 1.0, 0.0,   blue 0.0, 0.0, 1.0,
x = np.array([
    1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    -1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
], dtype=np.float32)
y = np.array([
    0.0, 2.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
], dtype=np.float32)
z = np.array([
    0.0, 0.0, 1.0, 0.9, 0.9, 0.0,
    0.0, 0.0, -1.0, 0.9, 0.9, 0.6,
], dtype=np.float32)
triangle = np.array([
    0.1, 0.9, -1.2, 1.0, 0.0, 0.0,
    1.2, 1.2, -1.2, 0.0, 1.0, 0.0,
    0.2, 0.2, -0.4, 0.0, 0.0, 1.0,
], dtype=np.float32)
# 声明几个数组的vbo,下面的均绑定在此vao中
vao = glGenVertexArrays(4)
# 声明几个数组的vbo,下面的均绑定在此vbo中
vbo = glGenBuffers(4)

glBindVertexArray(vao[0])
# 设置第一个vbo。即vbo[0]
glBindBuffer(GL_ARRAY_BUFFER, vbo[0])
glBufferData(GL_ARRAY_BUFFER, x.nbytes, x, GL_STATIC_DRAW)
glEnableVertexAttribArray(0)  # 准备设置glsl。即vertex_src中的layout(location = 0)的属性
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, x.itemsize * 6, ctypes.c_void_p(0))
glEnableVertexAttribArray(1)  # 准备设置glsl。即vertex_src中的layout(location = 1)的属性
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, x.itemsize * 6, ctypes.c_void_p(12))

glBindVertexArray(vao[1])
# 设置第二个vbo。即vbo[1]
glBindBuffer(GL_ARRAY_BUFFER, vbo[1])
glBufferData(GL_ARRAY_BUFFER, y.nbytes, y, GL_STATIC_DRAW)
glEnableVertexAttribArray(0)  # 准备设置glsl。即vertex_src中的layout(location = 0)的属性
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, y.itemsize * 6, ctypes.c_void_p(0))
glEnableVertexAttribArray(1)  # 准备设置glsl。即vertex_src中的layout(location = 1)的属性
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, y.itemsize * 6, ctypes.c_void_p(12))

glBindVertexArray(vao[2])
# 设置第三个vbo。即vbo[2]
glBindBuffer(GL_ARRAY_BUFFER, vbo[2])
glBufferData(GL_ARRAY_BUFFER, z.nbytes, z, GL_STATIC_DRAW)
glEnableVertexAttribArray(0)  # 准备设置glsl。即vertex_src中的layout(location = 0)的属性
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, z.itemsize * 6, ctypes.c_void_p(0))
glEnableVertexAttribArray(1)  # 准备设置glsl。即vertex_src中的layout(location = 1)的属性
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, z.itemsize * 6, ctypes.c_void_p(12))

glBindVertexArray(vao[3])
# 设置第三个vbo。即vbo[3]
glBindBuffer(GL_ARRAY_BUFFER, vbo[3])
glBufferData(GL_ARRAY_BUFFER, triangle.nbytes, triangle, GL_STATIC_DRAW)
glEnableVertexAttribArray(0)  # 准备设置glsl。即vertex_src中的layout(location = 0)的属性
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, triangle.itemsize * 6, ctypes.c_void_p(0))
glEnableVertexAttribArray(1)  # 准备设置glsl。即vertex_src中的layout(location = 1)的属性
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, triangle.itemsize * 6, ctypes.c_void_p(12))

vert_shader = glCreateShader(GL_VERTEX_SHADER)
program = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                         compileShader(fragment_src, GL_FRAGMENT_SHADER))
glUseProgram(program)
glEnable(GL_DEPTH_TEST)

glUniformMatrix4fv(glGetUniformLocation(program, 'projection'), 1, GL_FALSE, glm.value_ptr(Param.projection))
glUniformMatrix4fv(glGetUniformLocation(program, 'view'), 1, GL_FALSE, glm.value_ptr(Param.view))
# 设置模型坐标,和glUniformMatrix4fv不能写成一行，写成一行会无法展示，可能是因为用的指针地址
mat = glm.mat4()
glUniformMatrix4fv(glGetUniformLocation(program, 'model'), 1, GL_FALSE, glm.value_ptr(mat))


def th():
    global x, y, z
    while window_close:
        sleep(0.5)
        if Param.add:
            x = x.reshape(int(len(x) / 6), 6)
            x[:, 0] = x[:, 0] + 0.01
            x = x.flatten()
            y = y.reshape(int(len(y) / 6), 6)
            y[:, 0] = y[:, 0] - 0.01
            y = y.flatten()
            z = z.reshape(int(len(z) / 6), 6)
            z[:, 0] = z[:, 0] + 0.02
            z = z.flatten()
            if x[0] >= 2:
                Param.add = False
        else:
            x = x.reshape(int(len(x) / 6), 6)
            x[:, 0] = x[:, 0] - 0.01
            x = x.flatten()
            y = y.reshape(int(len(y) / 6), 6)
            y[:, 0] = y[:, 0] + 0.01
            y = y.flatten()
            z = z.reshape(int(len(z) / 6), 6)
            z[:, 0] = z[:, 0] - 0.02
            z = z.flatten()
            if x[0] <= -2:
                Param.add = True


window_close = True
threading.Thread(target=th, args=()).start()
while not glfw.window_should_close(window):
    glUniformMatrix4fv(glGetUniformLocation(program, 'projection'), 1, GL_FALSE,
                       glm.value_ptr(Param.projection))
    glUniformMatrix4fv(glGetUniformLocation(program, 'view'), 1, GL_FALSE, glm.value_ptr(Param.view))
    glClearColor(0.2, 0.3, 0.3, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0])
    glBufferData(GL_ARRAY_BUFFER, x.nbytes, x, GL_STATIC_DRAW)
    glBindVertexArray(vao[0])
    glDrawArrays(GL_LINES, 0, len(x))

    glBindBuffer(GL_ARRAY_BUFFER, vbo[1])
    glBufferData(GL_ARRAY_BUFFER, y.nbytes, y, GL_STATIC_DRAW)
    glBindVertexArray(vao[1])
    glDrawArrays(GL_LINES, 0, len(y))

    glBindBuffer(GL_ARRAY_BUFFER, vbo[2])
    glBufferData(GL_ARRAY_BUFFER, z.nbytes, z, GL_STATIC_DRAW)
    glBindVertexArray(vao[2])
    glDrawArrays(GL_LINES, 0, len(z))

    glBindVertexArray(vao[3])
    glDrawArrays(GL_TRIANGLES, 0, len(y))

    glfw.poll_events()
    glfw.swap_buffers(window)
window_close = False
glfw.terminate()
