import logging

import cv2
from ml_serving.utils import helpers

import numpy as np

from align.detect_align import detect_align
from warpgan import WarpGAN

PARAMS = {
    "model_dir": "",
    "scale": "1.0",
}

LOG = logging.getLogger(__name__)


def init_hook(ctx, **kwargs):
    PARAMS.update(kwargs)
    LOG.info('initialized hook with params: {}'.format(PARAMS))
    return _build_pipe(ctx)


def update_hook(ctx, **kwargs):
    PARAMS.update(kwargs)
    LOG.info('update hook with params: {}'.format(PARAMS))
    return _build_pipe(ctx)


def _build_pipe(ctx):
    network = WarpGAN()
    network.load_model(PARAMS.get('model_dir', './model'))
    styles = np.random.normal(0., 1., (1, network.input_style.shape[1].value))
    return network, styles


def process(inputs, ctx, **kwargs):

    frame, is_streaming = helpers.load_image(inputs, 'input')
    if frame is None:
        raise RuntimeError("Unable to read frame")

    network = ctx.global_ctx[0]
    styles = ctx.global_ctx[1]
    scale = 1.0

    frame = detect_align(frame)

    if frame is None:
        output = np.zeros((256, 256, 1), dtype="uint8")

    else:
        frame = (frame - 127.5) / 128.0

        images = np.tile(frame[None], [1, 1, 1, 1])
        scales = scale * np.ones((1))

        output = network.generate_BA(images, scales, 16, styles=styles)
        output = 0.5*output + 0.5

    if not is_streaming:
        output = output[:, :, ::-1]
        output = cv2.imencode('.jpg', output)[1].tostring()

    return {'output': output}
