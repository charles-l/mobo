import contextlib
import asyncio
import glfw
import skia
import os
from OpenGL import GL
import requests
import twitter_session
import av
from dataclasses import dataclass
from PIL import Image

import mobodb

db = mobodb.load_db(':memory:')
tsession = twitter_session.TwitterSession()
render_overlays = []
renderables = {}

text_paint = skia.Paint(AntiAlias=True, Color=skia.ColorGRAY)
text_font = skia.Font(None, 16, 1, 0)
paint = skia.Paint(
    AntiAlias=True,
    Style=skia.Paint.kStroke_Style,
    StrokeWidth=4,
    Color=skia.ColorRED
)

zoom = 1
invT = skia.Matrix()
char_queue = asyncio.Queue()
key_queue = asyncio.Queue()
draw_queue = asyncio.Queue()

context = None
canvas = None
surface = None


@contextlib.contextmanager
def glfw_window():
    if not glfw.init():
        raise RuntimeError('glfw.init() failed')
    glfw.window_hint(glfw.STENCIL_BITS, 8)
    w, h = 800, 600
    window = glfw.create_window(w, h, '', None, None)
    glfw.make_context_current(window)
    yield window
    glfw.terminate()


@contextlib.contextmanager
def skia_surface(window):
    context = skia.GrDirectContext.MakeGL()
    backend_render_target = skia.GrBackendRenderTarget(
        *glfw.get_window_size(window),
        0,  # sampleCnt
        0,  # stencilBits
        skia.GrGLFramebufferInfo(0, GL.GL_RGBA8))
    surface = skia.Surface.MakeFromBackendRenderTarget(
        context, backend_render_target, skia.kBottomLeft_GrSurfaceOrigin,
        skia.kRGBA_8888_ColorType, skia.ColorSpace.MakeSRGB())
    assert surface is not None
    yield surface
    context.abandonContext()


def load_img(path):
    if path.startswith('https://twitter.com'):
        components = path.removeprefix('https://twitter.com/').split('/')
        assert components[1] == 'status'
        tweet_id = components[2]
        print(tsession.get_tweet(tweet_id)['extended_entities']['media'])
        return load_img(tsession.get_tweet(tweet_id)['extended_entities']['media'][0]['media_url'])
    elif path.startswith('http://') or path.startswith('https://'):
        r = requests.get(path, stream=True)
        r.raw.decode_contents = True
        img = Image.open(r.raw)
        print('downloaded image from ', path)
        img = img.convert('RGBA')
        return skia.Image.frombytes(img.tobytes(), img.size, skia.kRGBA_8888_ColorType)
    elif path.startswith('/'):
        return skia.Image.open(path)
    else:
        assert False, f'Unknown path type for: {path}'


class Video:
    path: str
    container: av.container.input.InputContainer
    cur_frame: int
    cur_frame_data: skia.Image

    @staticmethod
    def _get_frame_data(frame_data: av.VideoFrame) -> skia.Image:
        f = frame_data.reformat(format='rgb32').to_ndarray()
        return skia.Image.fromarray(f,
                                    colorType=skia.ColorType.kBGRA_8888_ColorType)

    def __init__(self, path: str):
        self.path = str
        self.time = 0
        self.cur_frame = 0
        self.container = av.open(open(path, 'rb'))

        stream = self.container.streams.video[0]
        self.frames = stream.frames
        self.decode_stream = self.container.decode(stream)
        self.cur_frame_data = self._get_frame_data(next(self.decode_stream))
        self.framerate = stream.average_rate

    def render_frame(self) -> skia.Image:
        # FIXME: this function is hot garbage.
        # there's so much potential for things to end up out of sync
        # (e.g. frames/cur frame data/which stream we use, etc)
        # Order of operations is significant. Figure out how to get this info
        # directly from libav itself rather than duplicating the tracking here.

        assert self.cur_frame_data
        self.time = glfw.get_time()
        assert self.frames > 0
        goal_frame = int(self.time * self.framerate) % self.frames

        if self.cur_frame == goal_frame:
            # no need to do any work
            return self.cur_frame_data

        # XXX: this could potentially lock the thread if we can't catch up
        # with the goal_frame. (assuming we don't decode fast enough)
        while self.cur_frame != goal_frame:
            # handle wrap around
            self.cur_frame += 1
            if self.cur_frame > goal_frame:
                self.cur_frame = 0
                self.container.seek(0)
                self.decode_stream = self.container.decode(
                    self.container.streams.video[0])
            try:
                self.cur_frame_data = self._get_frame_data(
                    next(self.decode_stream))
            except StopIteration:
                print("YIKES, shouldn't have gotten here",
                      self.cur_frame, self.frames)

        return self.cur_frame_data

    def __del__(self):
        self.container.close()


def get_renderable(resource_path: str):
    if resource_path in renderables:
        # TODO: create a class for Image so we don't have to conditonally call
        # render_frame
        if isinstance(renderables[resource_path], Video):
            return renderables[resource_path].render_frame()
        else:
            return renderables[resource_path]
    if resource_path.endswith(('.png', '.jpg', '.jpeg')):
        renderables[resource_path] = load_img(resource_path)
        return renderables[resource_path]
    elif resource_path.endswith(('.mp4', '.webm')):
        renderables[resource_path] = Video(resource_path)
        return renderables[resource_path].render_frame()
    else:
        raise ValueError('Unknown file type for ', resource_path)


def render_frame(canvas, selected, dx, dy):
    # FIXME: this canvas.restore/canvas.save business is gross (and backwards)
    canvas.restore()

    global zoom, invT
    # canvas.translate(WIDTH//2, HEIGHT//2)
    canvas.scale(zoom, zoom)
    # canvas.translate(-(WIDTH//2), -(HEIGHT//2))
    canvas.translate(dx / canvas.getTotalMatrix().getScaleX(),
                     dy / canvas.getTotalMatrix().getScaleX())

    # reset transform
    zoom = 1

    # FIXME: a gross hack
    m = skia.Matrix()
    assert canvas.getTotalMatrix().invert(m)
    invT = m

    for r in db['images'].rows:
        img = get_renderable(r['image'])
        canvas.drawImage(img, r['x'], r['y'], paint)
        if r['id'] in selected:
            canvas.drawRect(
                skia.Rect(r['x'], r['y'], r['x'] + r['w'], r['y'] + r['h']), paint)

    canvas.save()
    canvas.resetMatrix()
    while render_overlays and (o := render_overlays.pop()):
        o()


def scroll_callback(window, xoffset, yoffset):
    global zoom
    zoom += yoffset * 0.05
    if zoom < 0.04:
        zoom = 0.04


def pan(window):
    while True:
        if (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) ==
                glfw.PRESS):
            lastx, lasty = glfw.get_cursor_pos(window)
            while (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) ==
                    glfw.PRESS):
                x, y = glfw.get_cursor_pos(window)
                yield (x - lastx, y - lasty)
                lastx, lasty = x, y
        yield (0, 0)


def to_global(p):
    '''Convert an (x, y) tuple to global space.'''
    x, y = p
    (gx, gy), = invT.mapPoints([skia.Point(x, y)])
    return gx, gy


def drag(window, selected):
    bindings = ','.join(['?'] * len(selected))
    start_pos = {r['id']: (r['x'], r['y']) for r in db['images'].rows_where(
        f'id in ({bindings})', list(selected))}
    startx, starty = to_global(glfw.get_cursor_pos(window))
    while True:
        endx, endy = to_global(glfw.get_cursor_pos(window))
        diffx, diffy = endx-startx, endy-starty
        for image_id in selected:
            db['images'].update(image_id,
                                {'x': start_pos[image_id][0] + diffx,
                                 'y': start_pos[image_id][1] + diffy})

        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) != glfw.PRESS:
            return

        yield


def select(window):
    selected = set()
    while True:
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            mosx, mosy = to_global(glfw.get_cursor_pos(window))
            selected = {x['id'] for x in db['images'].rows_where(
                '? between x and x + w and ? between y and y + h', [mosx, mosy])}
        yield selected


def drop_callback(window, paths):
    r = get_renderable(paths[0])
    x, y = to_global(glfw.get_cursor_pos(window))
    db['images'].insert({'image': paths[0],
                         'x': x,
                         'y': y,
                         'w': r.width(),
                         'h': r.height()})


def key_callback(window, key, scancode, action, mod):
    key_queue.put_nowait((key, action, mod))
    # TODO: move this stuff somewhere else and use the key_queue
    # TODO: add image support (using a proper clipboard management library)
    if mod == glfw.MOD_CONTROL and key == glfw.KEY_V and action == glfw.PRESS:
        s = glfw.get_clipboard_string(window).decode('utf-8')
        r = get_renderable(s)
        w, h = glfw.get_window_size(window)
        x, y = to_global((w//2, h//2))
        db['images'].insert({'image': s,
                             'x': x,
                             'y': y,
                             'w': r.width(),
                             'h': r.height()})


def char_callback(window, codepoint):
    global prompt
    char_queue.put_nowait(chr(codepoint))


@dataclass
class Bounds:
    x1: int
    y1: int
    x2: int
    y2: int

    def remove_from_top(self, height):
        self.y1 += height
        return Bounds(self.x1, self.y1 - height, self.x2, self.y1)

    def remove_from_bottom(self, height):
        self.y2 -= height
        return Bounds(self.x1, self.y2, self.x2, self.y2 + height)

    def __iter__(self):
        yield self.x1
        yield self.y1
        yield self.x2
        yield self.y2

    def to_rect(self):
        return skia.Rect(self.x1, self.y1, self.x2, self.y2)


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


async def select_box(window, select_data):
    prompt = ''
    select_i = 0

    def draw_overlay(data):
        bounds = Bounds(0, 0, *glfw.get_window_size(window))
        fuzzy_bounds = bounds.remove_from_bottom(200)
        canvas.drawRect(
            fuzzy_bounds.to_rect(),
            skia.Paint(Color=skia.ColorBLACK))

        prompt_bounds = fuzzy_bounds.remove_from_top(
            text_font.getSpacing())
        if prompt:
            blob = skia.TextBlob(prompt, text_font)
            canvas.drawTextBlob(
                blob, 5, prompt_bounds.y2 - text_font.getMetrics().fDescent, text_paint)

        for i, d in enumerate(data):
            row_bounds = fuzzy_bounds.remove_from_top(
                text_font.getSpacing())
            if i == select_i:
                canvas.drawRect(row_bounds.to_rect(),
                                skia.Paint(Color=skia.ColorBLUE))
            blob = skia.TextBlob(d, text_font)
            canvas.drawTextBlob(
                blob,
                5, row_bounds.y2 - text_font.getMetrics().fDescent,
                text_paint)

    while not glfw.get_key(window, glfw.KEY_ENTER) == glfw.PRESS:
        # FIXME: bad bad bad
        await asyncio.sleep(0.01)
        print(f'{prompt=}')
        while True:
            try:
                # FIXME: selectbox/handle_events stomp the key_queue for each other right now
                key, action, mod = key_queue.get_nowait()
                if key == glfw.KEY_BACKSPACE and action in (glfw.PRESS, glfw.REPEAT):
                    prompt = prompt[:-1]
                if key == glfw.KEY_UP and action == glfw.PRESS:
                    select_i -= 1
                if key == glfw.KEY_DOWN and action == glfw.PRESS:
                    select_i += 1
            except asyncio.QueueEmpty:
                break

        while True:
            try:
                char = char_queue.get_nowait()
                prompt += char
            except asyncio.QueueEmpty:
                break

        data = [d for d in select_data if prompt in d] if prompt else select_data
        select_i = -1 if select_i < -1 else select_i
        select_i = len(data) - 1 if select_i >= len(data) else select_i

        render_overlays.append(lambda: draw_overlay(data))
    if select_i == -1:
        return prompt
    else:
        return data[select_i]


async def handle_events(window):
    async def open_file():
        global db
        file = await select_box(window, os.listdir())
        db.conn.close()
        db = mobodb.load_db(file)

    async def handle_keys():
        key, action, mod = await key_queue.get()
        if (mod, key, action) == (glfw.MOD_CONTROL, glfw.KEY_O, glfw.PRESS):
            await open_file()

    while True:
        tasks = [asyncio.create_task(handle_keys())]
        done, pending = await asyncio.wait(tasks,
                                           return_when=asyncio.FIRST_COMPLETED)
        for p in pending:
            p.cancel()
        print("next")


async def draw_loop(window):
    global invT

    panner = pan(window)
    selector = select(window)
    click_drag = None
    while True:
        # wait for next redraw request
        # await draw_queue.get()
        await asyncio.sleep(0.001)
        canvas.clear(skia.ColorWHITE)

        dx, dy = next(panner)

        if click_drag is not None:
            try:
                next(click_drag)
            except StopIteration:
                click_drag = None
        else:
            selected = next(selector)
            if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
                click_drag = drag(window, selected)

        render_frame(canvas, selected, dx, dy)

        surface.flushAndSubmit()
        glfw.swap_buffers(window)


async def main():
    global context

    with glfw_window() as window:
        asyncio.create_task(handle_events(window))
        asyncio.create_task(draw_loop(window))
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        glfw.set_scroll_callback(window, scroll_callback)
        glfw.set_drop_callback(window, drop_callback)
        glfw.set_key_callback(window, key_callback)
        glfw.set_char_callback(window, char_callback)
        glfw.set_window_size_callback(window, resize_callback)

        context = skia.GrDirectContext.MakeGL()
        resize_callback(window, *glfw.get_window_size(window))

        while not glfw.window_should_close(window):
            glfw.poll_events()
            await asyncio.sleep(0.001)

        # be nice and cleanup
        context.abandonContext()


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        db = mobodb.load_db(sys.argv[1])
    asyncio.get_event_loop().run_until_complete(main())
