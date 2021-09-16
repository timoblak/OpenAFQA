import numpy as np


def blockproc(image, block_shape, fun, *fun_args, **fun_kwargs):
    """
    Processes image block by block, resulting in a new image of size image_height/block_size x image_width/block_size.
    Similar to matlab function blockproc. Result of each block is a single number.
    :param image: Grayscale fingerprint image
    :param block_shape: size of sub-block/sliding window to be processed
    :param fun: Function to be applied to a block
    :param fun_args: positional args to the function
    :param fun_kwargs: optional args to the function
    :return: processed image of same size as input
    """
    img_h, img_w = image.shape
    block_h, block_w = block_shape

    block_rows = int(np.ceil(img_h / block_h))
    block_cols = int(np.ceil(img_w / block_w))

    result = np.zeros(shape=(block_rows, block_cols))
    for i, r in enumerate(range(0, img_h, block_h)):
        for j, c in enumerate(range(0, img_w, block_w)):
            patch = image[r:min(r + block_w, img_h), c:min(c + block_w, img_w)]
            result[i, j] = fun(patch, *fun_args, **fun_kwargs)

    return result


def blockproc_inplace(image, block_shape, fun, *fun_args, **fun_kwargs):
    """
    Processes image IN PLACE block by block. Result of each block is a block of the same size.
    Intended for local enhancement/thresholding operations
    :param image: Grayscale fingerprint image
    :param block_shape: size of sub-block/sliding window to be processed
    :param fun: Function to be applied to a block
    :param fun_args: positional args to the function
    :param fun_kwargs: optional args to the function
    :return: processed image of same size as input
    """
    img_h, img_w = image.shape
    block_h, block_w = block_shape

    for i, r in enumerate(range(0, img_h, block_h)):
        for j, c in enumerate(range(0, img_w, block_w)):
            image[r:min(r + block_w, img_h), c:min(c + block_w, img_w)] = fun(image[r:min(r + block_w, img_h), c:min(c + block_w, img_w)], *fun_args, **fun_kwargs)
    return image
