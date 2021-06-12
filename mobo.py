import contextlib, glfw, skia
import traceback
import random
import functools
from OpenGL import GL
import sqlite_utils
import requests
import twitter_session
from PIL import Image

db = sqlite_utils.Database(':memory:')

tsession = twitter_session.TwitterSession()

db.create_table(
    'images', {'id': int, 'image': str, 'x': int, 'y': int, 'w': int, 'h': int},
    pk='id')

WIDTH, HEIGHT = 800, 600

@contextlib.contextmanager
def glfw_window():
    if not glfw.init():
        raise RuntimeError('glfw.init() failed')
    glfw.window_hint(glfw.STENCIL_BITS, 8)
    window = glfw.create_window(WIDTH, HEIGHT, '', None, None)
    glfw.make_context_current(window)
    yield window
    glfw.terminate()

@contextlib.contextmanager
def skia_surface(window):
    context = skia.GrDirectContext.MakeGL()
    backend_render_target = skia.GrBackendRenderTarget(
        WIDTH,
        HEIGHT,
        0,  # sampleCnt
        0,  # stencilBits
        skia.GrGLFramebufferInfo(0, GL.GL_RGBA8))
    surface = skia.Surface.MakeFromBackendRenderTarget(
        context, backend_render_target, skia.kBottomLeft_GrSurfaceOrigin,
        skia.kRGBA_8888_ColorType, skia.ColorSpace.MakeSRGB())
    assert surface is not None
    yield surface
    context.abandonContext()


@functools.lru_cache
def load_img(path):
    if path.startswith('https://twitter.com'):
        components = path.removeprefix('https://twitter.com/').split('/')
        assert components[1] == 'status'
        tweet_id = components[2]
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


paint = skia.Paint(
    AntiAlias=True,
    Style=skia.Paint.kStroke_Style,
    StrokeWidth=4,
    Color=skia.ColorRED,
)

def render_frame(canvas, selected):
    for r in db['images'].rows:
        img = load_img(r['image'])
        canvas.drawImage(img, r['x'], r['y'], paint)
        if r['id'] in selected:
            canvas.drawRect(skia.Rect(r['x'], r['y'], r['x'] + r['w'], r['y'] + r['h']), paint)

zoom = 1
def scroll_callback(window, xoffset, yoffset):
    global zoom
    zoom += yoffset * 0.05
    if zoom < 0.04:
        zoom = 0.04

def pan(window):
    pan = 0, 0
    while True:
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS:
            pan_start = glfw.get_cursor_pos(window)
            new_pan = 0, 0
            while glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS:
                xpos, ypos = glfw.get_cursor_pos(window)
                new_pan = (pan[0] + xpos - pan_start[0],
                           pan[1] + ypos - pan_start[1])
                yield new_pan
            pan = new_pan
        yield pan


def drag(window, selected):
    bindings = ','.join(['?'] * len(selected))
    start_pos = {r['id']: (r['x'], r['y']) for r in db['images'].rows_where(f'id in ({bindings})', list(selected))}
    update_query = f'update images set x = ?, y = ? where id in ({bindings})'
    startx, starty = glfw.get_cursor_pos(window)
    while True:
        endx, endy = glfw.get_cursor_pos(window)
        diffx, diffy = endx-startx, endy-starty
        for image_id in selected:
            db['images'].update(image_id, {'x': start_pos[image_id][0] + diffx,
                                           'y': start_pos[image_id][1] + diffy})

        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) != glfw.PRESS:
            return

        yield


invT = skia.Matrix()
def select(window):
    selected = set()
    while True:
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            mosx, mosy = glfw.get_cursor_pos(window)
            (gx, gy), = invT.mapPoints([skia.Point(mosx, mosy)])
            selected = {x['id'] for x in db['images'].rows_where('? between x and x + w and ? between y and y + h', [gx, gy])}
        yield selected


def drop_callback(window, paths):
    img = load_img(paths[0])
    db['images'].insert({'image': paths[0],
                         'x': -x, 'y': -y, 'w': img.width(), 'h': img.height()})

def key_callback(window, key, scancode, action, mod):
    if mod == glfw.MOD_CONTROL and key == glfw.KEY_V and action == glfw.PRESS:
        s = glfw.get_clipboard_string(window).decode('utf-8')
        img = load_img(s)
        db['images'].insert({'image': s,
                             'x': -x, 'y': -y, 'w': img.width(), 'h': img.height()})

with glfw_window() as window:
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    glfw.set_scroll_callback(window, scroll_callback)
    panner = pan(window)
    selector = select(window)
    glfw.set_drop_callback(window, drop_callback)
    glfw.set_key_callback(window, key_callback)

    click_drag = None
    while (glfw.get_key(window, glfw.KEY_ESCAPE) != glfw.PRESS
        and not glfw.window_should_close(window)):
        with skia_surface(window) as surface:
            with surface as canvas:
                canvas.clear(skia.ColorWHITE)
                x, y = next(panner)

                if click_drag is not None:
                    try:
                        next(click_drag)
                    except StopIteration:
                        click_drag = None
                else:
                    selected = next(selector)
                    if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
                        click_drag = drag(window, selected)

                canvas.translate(WIDTH//2, HEIGHT//2)
                canvas.scale(zoom, zoom)
                canvas.translate(-(WIDTH//2), -(HEIGHT//2))
                canvas.translate(x, y)

                # FIXME: a gross hack
                m = skia.Matrix()
                assert canvas.getTotalMatrix().invert(m)
                invT = m

                render_frame(canvas, selected)
            surface.flushAndSubmit()
            glfw.swap_buffers(window)

        glfw.wait_events()
