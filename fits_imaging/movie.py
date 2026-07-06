from pathlib import Path
from typing import Optional, Union
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_movie_from_images(image_cube: np.ndarray, output_path: Union[str, Path], fps: int = 8,
                             pixel_arcsec: float = 1.55, ffmpeg_path: Optional[str] = None):
    """Create an MP4 movie from a 3-D image cube: frame, y, x."""
    if image_cube.ndim != 3:
        raise ValueError("image_cube must have shape (nframe, ny, nx)")
    output_path = Path(output_path)
    if ffmpeg_path:
        matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path
    fig, ax = plt.subplots()
    im = ax.imshow(image_cube[0])
    ny, nx = image_cube.shape[1:]
    ax.set_title(f"{output_path.name}\n{pixel_arcsec*ny:.0f} x {pixel_arcsec*nx:.0f} arcsec")

    def update_frame(i):
        im.set_data(image_cube[i])
        return [im]

    ani = animation.FuncAnimation(fig, update_frame, frames=image_cube.shape[0], interval=1000 / fps, blit=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(output_path, writer="ffmpeg", fps=fps)
    plt.close(fig)
