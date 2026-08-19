import array

import moderngl
import moderngl_window as mglw


class Triangle(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Triangle Shader"
    window_size = (800, 600)
    aspect_ratio = None
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.program = self.ctx.program(
            vertex_shader="""
            #version 330

            in vec2 in_position;
            in vec3 in_color;

            out vec3 v_color;

            void main() {
                v_color = in_color;
                gl_Position = vec4(in_position, 0.0, 1.0);
            }
            """,
            fragment_shader="""
            #version 330

            in vec3 v_color;
            out vec4 f_color;

            void main() {
                f_color = vec4(v_color, 1.0);
            }
            """,
        )

        # x, y, r, g, b
        vertices = [
            0.0, 0.8, 1.0, 0.0, 0.0,
            -0.8, -0.8, 0.0, 1.0, 0.0,
            0.8, -0.8, 0.0, 0.0, 1.0,
        ]
        self.vbo = self.ctx.buffer(array.array("f", vertices).tobytes())
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, "2f 3f", "in_position", "in_color")],
        )

    def on_render(self, time, frame_time):
        self.ctx.clear(0.1, 0.1, 0.1)
        self.vao.render(moderngl.TRIANGLES)


def main():
    mglw.run_window_config(Triangle)


if __name__ == "__main__":
    main()
