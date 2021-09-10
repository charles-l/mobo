import contextlib
import asyncio
import glfw
import skia
from OpenGL import GL
from dataclasses import dataclass
from statechart import Machine, assign
import functools

machines = {}
renderers = {}

rect_machine = {
    'context': {'paint': skia.ColorGREEN},
    'initial': 'idle',
    'states': {
        'idle': {
            'on': {'enter':
                   {'target': 'hover',
                    'actions': [assign(paint=lambda: skia.ColorRED)]}},
            },
        'hover': {
            'on': {'leave':
                   {'target': 'idle',
                    'actions': [assign(paint=lambda: skia.ColorGREEN)]}},
            }
        }
    }

machines['rect'] = Machine(rect_machine)

def render_rect(rect, state):
    canvas.drawRect(skia.Rect(rect.x, rect.y, rect.x + rect.w, rect.y + rect.h),
                    skia.Paint(Color=state.context['paint']))

renderers['rect'] = render_rect


image_machine = {
    'context': {'path': '/home/nc/Downloads/IMG_0586.png'},
    'initial': 'idle',
    'states': {'idle': {}}
    }

machines['image'] = Machine(image_machine)
def render_image(rect, state):
    img = load_image(state.context['path'])
    canvas.drawImageRect(img,
                         skia.Rect(rect.x, rect.y, rect.x+rect.w, rect.y+rect.h)
                         )

renderers['image'] = render_image

@functools.lru_cache
def load_image(path: str):
    return skia.Image.open(path)

@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    type: str

    def hit_test(self, x, y):
        return (self.x <= x <= self.x + self.w and
                self.y <= y <= self.y + self.h)


@contextlib.contextmanager
def glfw_window():
    global arrow_cursor
    if not glfw.init():
        raise RuntimeError('glfw.init() failed')
    arrow_cursor = glfw.create_standard_cursor(glfw.ARROW_CURSOR)

    glfw.window_hint(glfw.STENCIL_BITS, 8)
    w, h = 800, 600
    window = glfw.create_window(w, h, '', None, None)
    glfw.make_context_current(window)
    yield window
    glfw.terminate()


states = {}
async def render_thread(window):
    for i, r in enumerate(get_renderables()):
        states[i] = machines[r.type].start()

    hit_test = set()
    while True:
        canvas.clear(skia.ColorWHITE)

        renderables = get_renderables()

        # do hit test
        x, y = glfw.get_cursor_pos(window)
        prev_hit_test = hit_test
        hit_test = set()
        for i, r in enumerate(renderables):
            if r.hit_test(x, y):
                hit_test.add(i)

        # pass renderables to renderer, with hit_test results
        for i, r in enumerate(renderables):
            # if it's an image, render the image
            # if it's a video, render the current frame
            #   in async, user a timer to switch frames
            # if it's text, render text into the box with proper line wrapping
            # somehow tie in focus here... -- maybe inputs only get sent to focus??
            # hmm.. how about the entire canvas is a state machine
            # and each individaul subcomponent gets a nested state machine that gets run on focus...
            if i in hit_test and i not in prev_hit_test:
                states[i] = machines[r.type].transition(states[i], 'enter')
            elif i not in hit_test and i in prev_hit_test:
                states[i] = machines[r.type].transition(states[i], 'leave')

            renderers[r.type](r, states[i])


        surface.flushAndSubmit()
        glfw.swap_buffers(window)
        await asyncio.sleep(0.001)


context = canvas = surface = None


def resize_callback(window, w, h):
    global context, canvas, surface
    backend_render_target = skia.GrBackendRenderTarget(
        w,
        h,
        0,  # sampleCnt
        0,  # stencilBits
        skia.GrGLFramebufferInfo(0, GL.GL_RGBA8))
    surface = skia.Surface.MakeFromBackendRenderTarget(
        context, backend_render_target, skia.kBottomLeft_GrSurfaceOrigin,
        skia.kRGBA_8888_ColorType, skia.ColorSpace.MakeSRGB())
    assert surface is not None

    canvas = surface.getCanvas()


def get_renderables():
    return [Rect(10, 10, 20, 200, 'rect'),
            Rect(40, 40, 800, 800, 'image')]


async def main():
    global context, canvas, surface

    with glfw_window() as window:

        context = skia.GrDirectContext.MakeGL()
        resize_callback(window, 800, 600)

        asyncio.create_task(render_thread(window))
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        while not glfw.window_should_close(window):
            glfw.poll_events()
            await asyncio.sleep(0.001)

        # be nice and cleanup
        context.abandonContext()


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())
