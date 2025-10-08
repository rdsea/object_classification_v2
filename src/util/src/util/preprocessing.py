import numpy as np


def _preprocess_numpy_input(x, data_format, mode):
    """Preprocesses a NumPy array encoding a batch of images.

    Args:
      x: Input array, 3D or 4D.
      data_format: Data format of the image array.
      mode: One of "caffe", "tf" or "torch".
        - caffe: will convert the images from RGB to BGR,
            then will zero-center each color channel with
            respect to the ImageNet dataset,
            without scaling.
        - tf: will scale pixels between -1 and 1,
            sample-wise.
        - torch: will scale pixels between 0 and 1 and then
            will normalize each channel with respect to the
            ImageNet dataset.

    Returns:
        Preprocessed Numpy array.
    """
    # if not issubclass(x.dtype.type, np.floating):
    #     x = x.astype(np.float32, copy=False)

    if mode == "tf":
        x = x / 127.5
        x -= 1.0
        return x
    elif mode == "torch":
        x /= 255.0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        if data_format == "channels_first":
            # 'RGB'->'BGR'
            if len(x.shape) == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        std = None

    # Zero-center by mean pixel
    if data_format == "channels_first":
        if len(x.shape) == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]
    return x


def preprocess_input(x, data_format="channels_last", mode="caffe"):
    """Preprocesses a tensor or Numpy array encoding a batch of images."""

    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(np.float32, copy=False)

    if mode == "raw":
        return x

    if mode not in {"caffe", "tf", "torch"}:
        raise ValueError(
            "Expected mode to be one of `caffe`, `tf` or `torch`. "
            f"Received: mode={mode}"
        )

    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "Expected data_format to be one of `channels_first` or "
            f"`channels_last`. Received: data_format={data_format}"
        )

    if isinstance(x, np.ndarray):
        return _preprocess_numpy_input(x, data_format=data_format, mode=mode)
    raise ValueError(
        "Expected data_format to be one of `channels_first` or "
        f"`channels_last`. Received: data_format={data_format}"
    )
