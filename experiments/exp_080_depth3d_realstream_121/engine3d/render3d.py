"""2.5D depth-mesh renderer: unproject a frame by its depth map and re-render it
from a moved virtual camera.

This is what makes a transition read as *3D* rather than as an overlay: parallax.
Foreground and background move at different rates because they genuinely sit at
different distances, so the eye reads camera motion through a scene instead of a
2D effect pasted over one.

Exactness at rest: at the identity camera the projection inverts the unprojection
regardless of depth, so a frame rendered with no camera motion reproduces itself.
That is what lets the conditioning buckets stay bit-accurate.
"""

from __future__ import annotations

import numpy as np

import moderngl

_VERTEX = """
#version 330 core
in vec2 aPix;                       // grid position in [0,1]^2, (0,0) = top-left of the image
uniform sampler2D uDepth;           // view-space z, positive, metres-ish
uniform float uTanHalfFov;
uniform float uAspect;
uniform mat4  uView;
uniform mat4  uProj;
uniform vec2  uShear;               // stereo-like swing about a convergence plane
uniform float uShearRef;            // convergence depth: this plane stays fixed
out vec2 vUV;
void main() {
    float z = texture(uDepth, vec2(aPix.x, 1.0 - aPix.y)).r;
    vec2 ndc = aPix * 2.0 - 1.0;
    vec3 P = vec3(ndc.x * uTanHalfFov * uAspect, ndc.y * uTanHalfFov, -1.0) * z;
    // Screen displacement is (world offset)/z, so a world offset of
    // shear*(1 - z/z_ref) yields a screen shift of shear*(1/z - 1/z_ref):
    // NEAR surfaces move most, the convergence plane is pinned, far surfaces
    // drift the other way — the real parallax law. Offsetting by (z - z_ref)
    // instead inverts it (far moves most), which the parallax metric caught as a
    // depth-flow correlation of -0.84.
    P.xy += uShear * (1.0 - z / max(uShearRef, 1e-3));
    gl_Position = uProj * uView * vec4(P, 1.0);
    vUV = aPix;
}
"""

# The alpha term is the disocclusion handling. Where the camera move stretches a
# triangle across a depth discontinuity, one texel is smeared over many pixels;
# |det J| of the UV derivatives detects exactly that and fades the smear out, so
# the other layer shows through instead of a rubber sheet. At the identity camera
# |det J| == 1 and alpha is exactly 1.
# The colour texture is RGBA: .rgb is the frame and .a carries a per-source-pixel
# mask. Because the mask is authored in *source* space and then warped with the
# geometry, anything driven by it (e.g. a dissolve evaluated at unprojected world
# positions) sticks to surfaces and parallaxes correctly instead of sliding over
# the image the way a screen-space mask does.
_FRAGMENT = """
#version 330 core
in vec2 vUV;
uniform sampler2D uColor;
uniform vec2  uTexSize;
uniform float uStretchLo;
uniform float uStretchHi;
out vec4 fragColor;
void main() {
    vec2 dx = dFdx(vUV), dy = dFdy(vUV);
    float texelsPerPixel = abs(dx.x * dy.y - dx.y * dy.x) * uTexSize.x * uTexSize.y;
    float a = smoothstep(uStretchLo, uStretchHi, texelsPerPixel);
    vec4 c = texture(uColor, vec2(vUV.x, 1.0 - vUV.y));
    fragColor = vec4(c.rgb, a * c.a);
}
"""


def perspective(fovy: float, aspect: float, znear: float = 0.05,
                zfar: float = 100.0) -> np.ndarray:
    f = 1.0 / np.tan(fovy / 2.0)
    m = np.zeros((4, 4), dtype="f4")
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2 * zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m


class MeshRenderer:
    """One EGL context + depth-tested FBO, reused for every layer render."""

    def __init__(self, width: int, height: int, step: int = 2):
        self.width, self.height, self.step = width, height, step
        self.ctx = moderngl.create_standalone_context(backend="egl")
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.color = self.ctx.texture((width, height), 4, dtype="f1")
        self.depth_rb = self.ctx.depth_renderbuffer((width, height))
        self.fbo = self.ctx.framebuffer([self.color], self.depth_rb)
        self.prog = self.ctx.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT)

        gw, gh = width // step + 1, height // step + 1
        xs = np.linspace(0.0, 1.0, gw, dtype="f4")
        ys = np.linspace(0.0, 1.0, gh, dtype="f4")
        grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2)
        i = np.arange(gw * gh).reshape(gh, gw)
        quads = np.stack([i[:-1, :-1], i[1:, :-1], i[:-1, 1:],
                          i[:-1, 1:], i[1:, :-1], i[1:, 1:]], -1)
        self._vbo = self.ctx.buffer(grid.astype("f4").tobytes())
        self._ibo = self.ctx.buffer(quads.astype("i4").tobytes())
        self._vao = self.ctx.vertex_array(
            self.prog, [(self._vbo, "2f", "aPix")], self._ibo)

        self._tex_color = self.ctx.texture((width, height), 4, dtype="f1")
        self._tex_color.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._tex_color.repeat_x = self._tex_color.repeat_y = False
        self._tex_depth = self.ctx.texture((width, height), 1, dtype="f4")
        self._tex_depth.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._tex_depth.repeat_x = self._tex_depth.repeat_y = False

    def renderer_name(self) -> str:
        return f"{self.ctx.info['GL_RENDERER']} | {self.ctx.info['GL_VERSION']}"

    def render(self, frame: np.ndarray, view_depth: np.ndarray, view: np.ndarray,
               *, fovy: float = 0.9, shear=(0.0, 0.0), shear_ref: float = 1.0,
               stretch_lo: float = 0.06, stretch_hi: float = 0.30) -> np.ndarray:
        """Render one layer. Returns (H, W, 4) uint8 — RGB plus a validity alpha.

        `frame` is (H, W, 3) or (H, W, 4); the 4th channel is a source-space mask
        multiplied into the output alpha (see the fragment shader).
        """
        if frame.shape[2] == 3:
            frame = np.concatenate(
                [frame, np.full(frame.shape[:2] + (1,), 255, np.uint8)], axis=2)
        self._tex_color.write(np.ascontiguousarray(frame).tobytes())
        self._tex_depth.write(np.ascontiguousarray(view_depth, dtype="f4").tobytes())
        self._tex_color.use(0)
        self._tex_depth.use(1)
        p = self.prog
        p["uColor"].value = 0
        p["uDepth"].value = 1
        aspect = self.width / self.height
        p["uAspect"].value = aspect
        p["uTanHalfFov"].value = float(np.tan(fovy / 2.0))
        p["uView"].write(np.ascontiguousarray(view.T, dtype="f4").tobytes())
        p["uProj"].write(np.ascontiguousarray(
            perspective(fovy, aspect).T, dtype="f4").tobytes())
        p["uShear"].value = (float(shear[0]), float(shear[1]))
        p["uShearRef"].value = float(shear_ref)
        p["uTexSize"].value = (float(self.width), float(self.height))
        p["uStretchLo"].value = float(stretch_lo)
        p["uStretchHi"].value = float(stretch_hi)

        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        self._vao.render(moderngl.TRIANGLES)
        raw = self.fbo.read(components=4, dtype="f1")
        img = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 4)
        return img[::-1]                       # GL returns bottom-up
