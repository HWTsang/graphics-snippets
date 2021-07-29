

class Param(object):
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
    """  # vec4(a_position, 1); 

    fragment_src = """
    # version 330
    in vec3 v_color;
    out vec4 out_color;
    void main()
    {
        out_color = vec4(v_color, 1.0);   
    }
    """
    width = 1280
    height = 720
    eye = glm.vec3(0, 0, 3) 
    target = glm.vec3(0, 0, 0)  
    up = glm.vec3(0, 1, 0)  
    near = 0.1
    far = 100
    radians = glm.radians(45)
    projection = glm.perspective(radians, width / height, near, far)
    view = glm.lookAt(eye, target, up)
    pan_start = None
    orbit_start = None
    pivot_world = None
    add = True


if not glfw.init():
    print("Cannot initialize GLFW")
    exit()
window = glfw.create_window(Param.width, Param.height, "Hello Triangle", None, None)
if not window:
    glfw.terminate()
glfw.window_hint(glfw.DOUBLEBUFFER, GL_TRUE)

glfw.make_context_current(window)

vao = glGenVertexArrays(1)
vbo = glGenBuffers(1)
ebo = glGenBuffers(1)

result = [
    np.array([0.220, 0.32, 0.32, 1.0, 0.0, 0.0, 0.32, 0.22, 0.32, 1.0, 0.0, 0.0,
              0.320, 0.22, 0.22, 1.0, 0.0, 0.0, 0.22, 0.32, 0.22, 1.0, 0.0, 0.0], dtype=np.float32),
    np.array([-0.955, -0.8557, -0.85554, 0.0, 1.0, 0.0, -0.8555, -0.955, -0.855, 0.0, 1.0, 0.0,
              -0.8557, -0.9573, -0.95553, 0.0, 1.0, 0.0, -0.955, -0.855, -0.955, 0.0, 1.0, 0.0], dtype=np.float32),
    np.array([-0.063320, 0.039, 0.036, 0.0, 0.0, 1.0, 0.036, -0.0630, 0.03679, 0.0, 0.0, 1.0,
              0.036679, -0.063, -0.0630, 0.0, 0.0, 1.0, -0.060, 0.03679, -0.063, 0.0, 0.0, 1.0], dtype=np.float32)
]
ins = [np.array([2, 3, 1, 3, 0, 1], dtype=np.uint32), np.array([2, 3, 1, 3, 0, 1], dtype=np.uint32),
       np.array([2, 3, 1, 3, 0, 1], dtype=np.uint32)]

glBindVertexArray(vao)
glBindBuffer(GL_ARRAY_BUFFER, vbo)
result_ = sum(len(i) for i in result) * 4
glBufferData(GL_ARRAY_BUFFER, result_, None, GL_STATIC_DRAW)
offset = 0
for i in result:
    arr = (ctypes.c_float * i.nbytes)(*i)
    glBufferSubData(GL_ARRAY_BUFFER, offset, i.nbytes, arr)
    offset += i.nbytes

glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
ins_ = sum(len(i) for i in ins) * 4
glBufferData(GL_ELEMENT_ARRAY_BUFFER, ins_, None, GL_STATIC_DRAW)
offset = 0
for i in ins:
    arr = (ctypes.c_float * i.nbytes)(*i)
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, offset, i.nbytes, arr)
    offset += i.nbytes

glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

program = compileProgram(compileShader(Param.vertex_src, GL_VERTEX_SHADER),
                         compileShader(Param.fragment_src, GL_FRAGMENT_SHADER))
glUseProgram(program)
glEnable(GL_DEPTH_TEST)

glUniformMatrix4fv(glGetUniformLocation(program, 'projection'), 1, GL_FALSE, glm.value_ptr(Param.projection))
glUniformMatrix4fv(glGetUniformLocation(program, 'view'), 1, GL_FALSE, glm.value_ptr(Param.view))
mat = glm.mat4()
glUniformMatrix4fv(glGetUniformLocation(program, 'model'), 1, GL_FALSE, glm.value_ptr(mat))

while not glfw.window_should_close(window):
    glUniformMatrix4fv(glGetUniformLocation(program, 'projection'), 1, GL_FALSE,
                       glm.value_ptr(Param.projection))
    glUniformMatrix4fv(glGetUniformLocation(program, 'view'), 1, GL_FALSE, glm.value_ptr(Param.view))
    glClearColor(0.2, 0.3, 0.3, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glBindVertexArray(vao)
    glDrawElements(GL_TRIANGLE_STRIP, sum([len(i) for i in ins]), GL_UNSIGNED_INT, None)

    glfw.poll_events()
    glfw.swap_buffers(window)
glfw.terminate()
