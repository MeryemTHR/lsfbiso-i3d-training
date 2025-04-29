import numpy as np
import PIL

# Code copied from : https://github.com/hassony2/torch_videovision/blob/master/torchvideotransforms/functional.py


def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h : min_h + h, min_w : min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip]
    else:
        raise TypeError(
            "Expected numpy.ndarray or PIL.Image"
            + "but got list of {0}".format(type(clip[0]))
        )
    return cropped


def resize_clip(clip, size, interpolation="bilinear"):
    # Handle empty clips
    if len(clip) == 0:
        # Return a dummy clip of the target size
        if isinstance(size, tuple):
            h, w = size
        else:
            h, w = size, size
        return np.zeros((1, h, w, 3), dtype=np.float32)
    
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, tuple):
            im_h, im_w = size
        else:
            im_w = im_h = size
        
        # Check clip dimensions
        clip_h, clip_w = clip[0].shape[:2]
        
        if clip_h == im_h and clip_w == im_w:
            return clip
            
        if interpolation == "bilinear":
            np_inter = 1
        else:
            np_inter = 0
            
        # Reshape to target size
        resized = np.array(
            [
                np.array(
                    PIL.Image.fromarray(img.astype(np.uint8)).resize(
                        (im_w, im_h), np_inter
                    )
                ).astype(float)
                / 255.0
                for img in clip
            ]
        )

        return resized
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, tuple):
            im_h, im_w = size
        else:
            im_w = im_h = size
        return [img.resize((im_w, im_h)) for img in clip]
    else:
        raise TypeError(
            "Expected numpy.ndarray or PIL.Image"
            + "but got list of {0}".format(type(clip[0]))
        )
