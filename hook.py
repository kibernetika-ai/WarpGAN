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

    LOG.info("process incoming")

    frame, is_streaming = helpers.load_image(inputs, 'input')
    if frame is None:
        raise RuntimeError("Unable to read frame")

    LOG.info("input frame size: {}".format(frame.shape))

    network = ctx.global_ctx[0]
    styles = ctx.global_ctx[1]
    scale = 1.0

    frame = detect_align(frame)
    LOG.info("aligned frame size: {}".format(frame.shape))

    if frame is None:
        output = np.zeros((256, 256, 1), dtype="uint8")
        LOG.info("no aligned image")

    else:
        frame = (frame - 127.5) / 128.0
        LOG.info("aligned image exists")

        images = np.tile(frame[None], [1, 1, 1, 1])
        scales = scale * np.ones(1)

        output = network.generate_BA(images, scales, 16, styles=styles)
        output = 0.5*output + 0.5

        output = (output[0] * 256).astype('uint8')

    LOG.info("output frame size: {}".format(output.shape))
    if not is_streaming:
        _, buf = cv2.imencode('.jpg', output[:, :, ::-1])
        output = buf.tostring()

    return {'output': output}
