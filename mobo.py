import contextlib
import asyncio
import glfw
import skia
import math
import io
import os
import re
import hashlib
from urllib.parse import urlparse
from OpenGL import GL
import requests
import twitter_session
import av
from dataclasses import dataclass
import PIL.Image
from typing import BinaryIO, Dict, Union
from sqlite_utils.db import NotFoundError

import mobodb

db = mobodb.load_db(':memory:')
tsession = twitter_session.TwitterSession()
render_overlays = []
renderables: Dict[str, Union['Video', 'Image']] = {}

text_paint = skia.Paint(AntiAlias=True, Color=skia.ColorGRAY)
text_font = skia.Font(None, 16, 1, 0)
paint = skia.Paint(
    AntiAlias=True,
    Style=skia.Paint.kStroke_Style,
    StrokeWidth=4,
    Color=skia.ColorRED
)

zoom = 1
origin = (0, 0)
invT = skia.Matrix()

# TODO: figure out a better way of message passing than creating a new
# queue for each message type
char_queue = asyncio.Queue()
key_queue = asyncio.Queue()
draw_queue = asyncio.Queue()
scroll_queue = asyncio.Queue()

context = None
canvas = None
surface = None
arrow_cursor = None


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


class Video:
    container: av.container.input.InputContainer
    cur_frame: int
    cur_frame_data: skia.Image

    @staticmethod
    def _get_frame_data(frame_data: av.VideoFrame) -> skia.Image:
        f = frame_data.reformat(format='rgb32').to_ndarray()
        return skia.Image.fromarray(f,
                                    colorType=skia.ColorType.kBGRA_8888_ColorType)

    def __init__(self, f: BinaryIO):
        self.time = 0
        self.cur_frame = 0
        self.container = av.open(f)

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


class Image:
    image: skia.Image

    def __init__(self, f: BinaryIO):
        img = PIL.Image.open(f)
        img = img.convert('RGBA')
        self.image = skia.Image.frombytes(
            img.tobytes(), img.size, skia.kRGBA_8888_ColorType)

    def render_frame(self) -> skia.Image:
        return self.image


def _get_type(path):
    # TODO: use libmagic to extract file type
    if path.endswith(('.mp4', '.webm')):
        return 'video'
    elif path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        return 'image'
    raise Exception('unknown extension for', path)


def _get_or_insert_asset(source_path: str, f: BinaryIO):
    # potential TODO: stream rather than loading whole buffer into memory
    # although it will still be two passes anyway...
    buf = f.read()
    m = hashlib.sha256()
    m.update(buf)
    try:
        db['assets'].get(m.hexdigest())
    except NotFoundError:
        db['assets'].insert({
            'id': m.hexdigest(),
            'blob': buf,
            'source': source_path,
            'type': _get_type(source_path)})
    return m.hexdigest()


def import_or_load_asset(resource_path: str):
    # early exit case: db already has the asset imported
    matching_assets = list(db['assets'].rows_where('source = ?', (resource_path,)))
    if matching_assets:
        print(resource_path, 'is already imported, skipping loading from path')
        # I don't think it's possible to internally reach a state where duplicate
        # resources for the same path would be loaded, but assert it for now.
        #
        # This might change in the future if external resources can be imported
        # multiple times over time (e.g. to version control differences).
        assert len(matching_assets) == 1
        return matching_assets[0]['id']

    if re.match(r'^https?://', resource_path):
        # parse out resource
        url = urlparse(resource_path)
        download_path = resource_path
        if url.netloc == 'twitter.com':
            components = url.path.removeprefix('/').split('/')
            if components[1] != 'status':
                raise ValueError("Can't handle twitter path", url.path, components[1])
            tweet_id = components[2]
            # TODO: add dropdown selection for this:
            all_media = tsession.get_tweet(tweet_id)['extended_entities']['media']

            print(all_media)
            media = all_media[-1]
            download_path = None

            # try to fetch video first
            if 'video_info' in media:
                for v in media['video_info']['variants']:
                    if v['content_type'] == 'video/mp4':
                        # strip off parameters because twitter adds weird stuff
                        u = urlparse(v['url'])
                        download_path = u.scheme + '://' + u.netloc + u.path
                        break

            # fall back to thumbnail
            if download_path is None:
                download_path = media['media_url']

            print('downloading ', download_path)

        # generic downloader
        r = requests.get(download_path, stream=True)
        r.raw.decode_contents = True
        path, f = download_path, r.raw
    else:
        if not os.path.exists(resource_path):
            raise Exception("can't import", resource_path)
        path, f = resource_path, open(resource_path, 'rb')

    try:
        return _get_or_insert_asset(path, f)
    finally:
        f.close()


def get_renderable(asset_id: str):
    if asset_id in renderables:  # already fetched
        return renderables[asset_id].render_frame()

    asset = db['assets'].get(asset_id)
    blobf = io.BytesIO(asset['blob'])

    if asset['type'] == 'image':
        renderables[asset_id] = Image(blobf)
    elif asset['type'] == 'video':
        renderables[asset_id] = Video(blobf)
    else:
        raise ValueError('Unknown file type for ', asset_id)
    return renderables[asset_id].render_frame()


def scroll_callback(window, xoffset, yoffset):
    scroll_queue.put_nowait((xoffset, yoffset))


def pan(window):
    def mouse_trigger():
        return glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
    def space_trigger():
        return glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS
    while True:
        trigger = None
        if mouse_trigger():
            trigger = mouse_trigger
        if space_trigger():
            trigger = space_trigger

        if trigger:
            glfw.set_cursor(window, arrow_cursor)
            lastx, lasty = glfw.get_cursor_pos(window)
            while trigger():
                x, y = glfw.get_cursor_pos(window)
                yield (x - lastx, y - lasty)
                lastx, lasty = x, y
            glfw.set_cursor(window, None)
        yield (0, 0)


def to_global(p):
    '''Convert an (x, y) tuple to global space.'''
    x, y = p
    (gx, gy), = invT.mapPoints([skia.Point(x, y)])
    return gx, gy


def drag_pos(window, end_trigger, selected):
    bindings = ','.join(['?'] * len(selected))
    start_pos = {r['id']: (r['x'], r['y']) for r in db['images'].rows_where(
        f'id in ({bindings})', list(selected))}
    for diffx, diffy in drag(window, end_trigger):
        for image_id in selected:
            db['images'].update(image_id,
                                {'x': start_pos[image_id][0] + diffx,
                                 'y': start_pos[image_id][1] + diffy})
        yield


def drag_scale(window, end_trigger, selected):
    bindings = ','.join(['?'] * len(selected))
    start_scale = {r['id']: r['scale'] for r in db['images'].rows_where(
        f'id in ({bindings})', list(selected))}
    for diffx, diffy in drag(window, end_trigger):
        for image_id in selected:
            db['images'].update(image_id,
                                {'scale': max(start_scale[image_id] + (diffx+diffy)/1000, 0.01)})
        yield



def drag(window, end_trigger):
    startx, starty = to_global(glfw.get_cursor_pos(window))
    while True:
        endx, endy = to_global(glfw.get_cursor_pos(window))
        diffx, diffy = endx-startx, endy-starty
        yield diffx, diffy

        if end_trigger():
            return


def select(window):
    selected = set()
    while True:
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            mosx, mosy = to_global(glfw.get_cursor_pos(window))
            new_selected = {x[0] for x in db.execute(
                'select id, z from images where ? between x and x + w*scale and ? between y and y + h*scale order by z desc limit 1', [mosx, mosy]).fetchall()}
            if len(new_selected) == 1 and next(iter(new_selected)) in selected:
                yield selected
                continue
            if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
                selected |= new_selected
            else:
                selected = new_selected
        yield selected


def drop_callback(window, paths):
    asset_id = import_or_load_asset(paths[0])
    r = get_renderable(asset_id)
    x, y = to_global(glfw.get_cursor_pos(window))
    db['images'].insert({'asset_id': asset_id,
                         'x': x,
                         'y': y,
                         'z': db.execute('select ifnull(max(z),0)+1 from images').fetchone()[0],
                         'w': r.width(),
                         'h': r.height()})


def key_callback(window, key, scancode, action, mod):
    key_queue.put_nowait((key, action, mod))
    # TODO: move this stuff somewhere else and use the key_queue
    # TODO: add image support (using a proper clipboard management library)
    if mod == glfw.MOD_CONTROL and key == glfw.KEY_V and action == glfw.PRESS:
        s = glfw.get_clipboard_string(window).decode('utf-8')
        asset_id = import_or_load_asset(s)
        r = get_renderable(asset_id)
        w, h = glfw.get_window_size(window)
        x, y = to_global((w//2, h//2))
        db['images'].insert({'asset_id': asset_id,
                             'x': x,
                             'y': y,
                             'z': db.execute('select ifnull(max(z),0)+1 from images').fetchone()[0],
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
    canvas.translate(origin[0], origin[1])
    canvas.scale(zoom, zoom)


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


async def draw_loop(window):
    global invT, zoom, origin

    panner = pan(window)
    selector = select(window)
    click_drag = None
    while True:
        # wait for next redraw request
        # await draw_queue.get()
        await asyncio.sleep(0.001)
        canvas.clear(skia.ColorWHITE)

        dx, dy = next(panner)
        origin = (origin[0] + dx, origin[1] + dy)

        if click_drag is not None:
            try:
                next(click_drag)
            except StopIteration:
                click_drag = None
        else:
            selected = next(selector)

            if selected:
                top_z = db.execute('select max(z) from images').fetchone()[0]
                sel_template = ','.join(['?'] * len(selected))
                min_selected_z = db.execute(f'select min(z) from images where id in ({sel_template})', selected).fetchone()[0]
                for i, img in enumerate(db['images'].rows_where(f'z > ? and id not in ({sel_template})', (min_selected_z, *selected))):
                    db['images'].update(img['id'], {'z': min_selected_z + i})

                for i, img in enumerate(db['images'].rows_where(f'id in ({sel_template})', selected, order_by='z')):
                    db['images'].update(img['id'], {'z': top_z - len(selected) + i + 1})

            if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
                click_drag = drag_pos(window, lambda: glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) != glfw.PRESS,
                                      selected)

            if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
                click_drag = drag_scale(window, lambda: glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) != glfw.PRESS,
                                        selected)

            if glfw.get_key(window, glfw.KEY_DELETE) == glfw.PRESS or glfw.get_key(window, glfw.KEY_BACKSPACE) == glfw.PRESS:
                for image_id in list(selected):
                    db['images'].delete(image_id)
                    selected.remove(image_id)

        canvas.restore()

        try:
            dsy = scroll_queue.get_nowait()[1] * 0.05
            zoom += (dsy * zoom)
            dsy += 1
        except asyncio.QueueEmpty:
            dsy = 1

        # canvas.translate(WIDTH//2, HEIGHT//2)
        canvas.scale(dsy, dsy)
        # canvas.translate(-(WIDTH//2), -(HEIGHT//2))
        canvas.translate(dx / canvas.getTotalMatrix().getScaleX(),
                         dy / canvas.getTotalMatrix().getScaleX())

        # FIXME: a gross hack
        m = skia.Matrix()
        assert canvas.getTotalMatrix().invert(m)
        invT = m

        for r in db['images'].rows_where(order_by='z'):
            img = get_renderable(r['asset_id'])

            x, y, w, h = r['x'], r['y'], r['w'] * r['scale'], r['h'] * r['scale']

            canvas.drawImageRect(img,
                                 skia.Rect(0, 0, r['w'], r['h']),
                                 skia.Rect(x, y, x+w, y+h), paint)
            if r['id'] in selected:
                canvas.drawRect(
                    skia.Rect(x, y, x+w, y+h), paint)

        canvas.save()
        canvas.resetMatrix()
        while render_overlays and (o := render_overlays.pop()):
            o()

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
