import argparse

import cv2
import imageio
import numpy as np
from warpgan import WarpGAN
from align.detect_align import detect_align


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="The path to the pretrained model",
                        type=str, default="pretrained/warpgan_pretrained")
    parser.add_argument("--input", help="The path to the aligned image",
                        type=str, default=None)
    parser.add_argument("--output",
                        help="The prefix path to the output file, subfix will be added for different styles.",
                        type=str, default=None)
    parser.add_argument("--num_styles", help="The number of images to generate with different styles",
                        type=int, default=5)
    parser.add_argument("--scale", help="The path to the input directory",
                        type=float, default=1.0)
    parser.add_argument("--aligned", help="Set true if the input face is already normalized",
                        action='store_true')
    return parser.parse_args()


def init_and_load_model(model_dir):
    network = WarpGAN()
    network.load_model(model_dir)
    return network


def process(network, img, aligned=False, styles=None, scale=1.):

    if not aligned:
        img = detect_align(img)
        if img is None:
            return None

    img = (img - 127.5) / 128.0

    images = np.tile(img[None], [num_styles, 1, 1, 1])
    scales = scale * np.ones((num_styles))

    output = network.generate_BA(images, scales, 16, styles=styles)
    output = 0.5*output + 0.5

    return output


if __name__ == '__main__':

    args = parse_args()
    network = init_and_load_model(args.model_dir)
    is_cam = args.input is None
    num_styles = 1 if is_cam else args.num_styles
    styles = np.random.normal(0., 1., (num_styles, network.input_style.shape[1].value))

    if is_cam:

        cam = cv2.VideoCapture(0)
        while True:
            ret, frame_in = cam.read()
            frame_in = frame_in[:, :, ::-1]

            output = process(network, frame_in, aligned=args.aligned, styles=styles, scale=args.scale)
            if output is None:
                output = np.zeros((256, 256, 1), dtype="uint8")
                print("skipped")
            else:
                output = output[0]

            cv2.imshow('webcam', output)

            ch = 0xFF & cv2.waitKey(1)
            if ch in [27, ord('q')]:
                break

    else:

        img = imageio.imread(args.input, pilmode='RGB')

        output = process(
            network, img,
            aligned=args.aligned, styles=styles, scale=args.scale,
        )

        if output is not None:
            for i in range(args.num_styles):
                imageio.imwrite(args.output + '_{}.jpg'.format(i), output[i])
