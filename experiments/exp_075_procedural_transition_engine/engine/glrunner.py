"""Headless OpenGL runner for gl-transitions shaders (moderngl + EGL).

Works with no display and no GPU: the cluster's Mesa stack exposes a GL 4.5 core
profile through EGL on llvmpipe, so operator rendering can run on ordinary CPU
nodes and never competes with training for GPUs. On a node with the NVIDIA
driver visible, the same code picks up `libEGL_nvidia` and runs on the GPU.
"""

from __future__ import annotations

import numpy as np

import moderngl

_VERTEX = """
#version 330 core
in vec2 _pos;
out vec2 _uv;
void main() {
    _uv = (_pos + 1.0) * 0.5;
    gl_Position = vec4(_pos, 0.0, 1.0);
}
"""

# gl-transitions shaders are authored against a WebGL-ish preamble: they may call
# getFromColor/getToColor, and read `progress` and `ratio`. `texture2D` is aliased
# because a handful of shaders still use the GLSL-ES spelling.
_FRAGMENT_HEADER = """
#version 330 core
#define texture2D texture
uniform sampler2D _fromTex;
uniform sampler2D _toTex;
uniform float progress;
uniform float ratio;
in vec2 _uv;
out vec4 _fragColor;
vec4 getFromColor(vec2 uv) { return texture(_fromTex, vec2(uv.x, 1.0 - uv.y)); }
vec4 getToColor(vec2 uv)   { return texture(_toTex,   vec2(uv.x, 1.0 - uv.y)); }
"""

_FRAGMENT_FOOTER = """
void main() { _fragColor = transition(_uv); }
"""


class ShaderCompileError(RuntimeError):
    pass


class GLRunner:
    """One EGL context + framebuffer, reused across every operator render."""

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.ctx = moderngl.create_standalone_context(backend="egl")
        self.fbo = self.ctx.simple_framebuffer((width, height), components=4)
        self.fbo.use()
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self._vbo = self.ctx.buffer(quad.tobytes())
        self._tex_from = self._make_tex()
        self._tex_to = self._make_tex()
        self._tex_aux = None
        self._prog_cache: dict[str, moderngl.Program] = {}

    # -- resources ---------------------------------------------------------
    def _make_tex(self):
        t = self.ctx.texture((self.width, self.height), 3, dtype="f1")
        t.repeat_x = t.repeat_y = False        # clamp-to-edge, as gl-transitions assumes
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return t

    def renderer_name(self) -> str:
        return f"{self.ctx.info['GL_RENDERER']} | {self.ctx.info['GL_VERSION']}"

    def program(self, shader_name: str, shader_source: str) -> moderngl.Program:
        if shader_name not in self._prog_cache:
            frag = _FRAGMENT_HEADER + shader_source + _FRAGMENT_FOOTER
            try:
                self._prog_cache[shader_name] = self.ctx.program(
                    vertex_shader=_VERTEX, fragment_shader=frag
                )
            except Exception as exc:                       # compile / link failure
                raise ShaderCompileError(f"{shader_name}: {exc}") from exc
        return self._prog_cache[shader_name]

    def set_aux_map(self, arr: np.ndarray) -> None:
        """Upload the auxiliary sampler used by luma/displacement shaders."""
        if self._tex_aux is not None:
            self._tex_aux.release()
        h, w = arr.shape[:2]
        self._tex_aux = self.ctx.texture((w, h), 3, arr.astype("u1").tobytes())
        self._tex_aux.repeat_x = self._tex_aux.repeat_y = False
        self._tex_aux.filter = (moderngl.LINEAR, moderngl.LINEAR)

    # -- rendering ---------------------------------------------------------
    @staticmethod
    def _assign(prog: moderngl.Program, name: str, value) -> bool:
        member = prog.get(name, None)
        if member is None:                     # optimised out by the compiler
            return False
        try:
            member.value = value
            return True
        except Exception:
            return False

    def render(self, prog, frame_from: np.ndarray, frame_to: np.ndarray,
               progress: float, params: dict, aux_uniform: str | None = None
               ) -> np.ndarray:
        """Render one transition frame. Inputs/outputs are HxWx3 uint8, row 0 = top."""
        self._tex_from.write(np.ascontiguousarray(frame_from).tobytes())
        self._tex_to.write(np.ascontiguousarray(frame_to).tobytes())
        self._tex_from.use(0)
        self._tex_to.use(1)
        self._assign(prog, "_fromTex", 0)
        self._assign(prog, "_toTex", 1)
        self._assign(prog, "progress", float(progress))
        self._assign(prog, "ratio", self.width / self.height)
        if aux_uniform and self._tex_aux is not None:
            self._tex_aux.use(2)
            self._assign(prog, aux_uniform, 2)
        for k, v in params.items():
            self._assign(prog, k, v)

        vao = self.ctx.vertex_array(prog, [(self._vbo, "2f", "_pos")])
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        vao.render(moderngl.TRIANGLE_STRIP)
        vao.release()

        raw = self.fbo.read(components=3, dtype="f1")
        img = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
        return img[::-1]                       # GL returns bottom-up
