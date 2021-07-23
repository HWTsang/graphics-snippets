import glfw
import numpy as np
import pyrr
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram

vertex_src = """
# version 330
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;
out vec3 v_color;
void main()
{
    gl_Position = vec4(a_position, 1.0);
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

# geometry to use. these are 3 xyz points (9 floats total) to make a triangle */
vertices = [0.0, 0.5, 0.0, 0.5, -0.5, 0.0, -0.5, -0.5, 0.0, 0.4, -0.2, 0.5]
vertices = np.array(vertices, dtype=np.float32)
vertices_color = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
vertices_color = np.array(vertices_color, dtype=np.float32)

if not glfw.init():
    print("Cannot initialize GLFW")
    exit()

window = glfw.create_window(640, 480, "Hello Triangle", None, None)
if not window:
    glfw.terminate()

glfw.make_context_current(window)
# glfw.set_cursor_pos_callback(window,)
# glfw.set_scroll_callback(window,)

# 声明1个数组的vbo,下面的均绑定在此vao中
vao = glGenVertexArrays(1)
glBindVertexArray(vao)

# 声明两个数组的vbo,下面的均绑定在此vbo中
vbo = glGenBuffers(2)

# 设置第一个vbo。即vbo[0]
glBindBuffer(GL_ARRAY_BUFFER, vbo[0])
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
glEnableVertexAttribArray(0)  # 准备设置glsl。即vertex_src中的layout(location = 0)的属性
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

# 设置第一个vbo。即vbo[1]
glBindBuffer(GL_ARRAY_BUFFER, vbo[1])
glBufferData(GL_ARRAY_BUFFER, vertices_color.nbytes, vertices_color, GL_STATIC_DRAW)
glEnableVertexAttribArray(1)  # 准备设置glsl。即vertex_src中的layout(location = 1)的属性
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

vert_shader = glCreateShader(GL_VERTEX_SHADER)
shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))
glUseProgram(shader)

while not glfw.window_should_close(window):
    # 声明要画的数据
    glBindVertexArray(vao)

    # 画3个点
    glDrawArrays(GL_LINE_STRIP, 0, 4)

    glfw.poll_events()
    glfw.swap_buffers(window)

glfw.terminate()
