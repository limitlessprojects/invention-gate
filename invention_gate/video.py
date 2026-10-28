"""Video and mask reading. mp4/avi/mov via opencv, .npy stacks, .png masks."""
import os
import numpy as np


def read_frames(path, max_frames=None):
    """Return a float-stack list of frames from a video file or .npy array."""
    path = str(path)
    if path.endswith(".npy"):
        arr = np.load(path)
        frames = [f for f in arr]
    elif os.path.isdir(path):
        from PIL import Image  # optional
        names = sorted(os.listdir(path))
        frames = [np.asarray(Image.open(os.path.join(path, n))) for n in names]
    else:
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"cannot open {path}")
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and len(frames) >= max_frames):
                break
            frames.append(frame[:, :, ::-1])  # BGR -> RGB
        cap.release()
    if max_frames:
        frames = frames[:max_frames]
    return frames
