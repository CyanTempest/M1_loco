"""Plot the actual seeded training heightfield, using its public generator."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate
from matplotlib.ticker import MaxNLocator
from omegaconf import OmegaConf
from unilab.terrains import TerrainGenerator

from .assets import PACKAGE_ROOT


def main():
    cfg = OmegaConf.load(Path(__file__).with_name("terrain.yaml"))
    generator = TerrainGenerator(instantiate(cfg.terrain.generator, _convert_="object"))
    terrain = generator.generate()
    names = list(cfg.terrain.generator.sub_terrains)
    grid = terrain.heights_yx
    border = round(cfg.terrain.generator.border_width / terrain.horizontal_scale)
    tile = round(cfg.terrain.generator.size[0] / terrain.horizontal_scale)
    level = 2
    fig = plt.figure(figsize=(14, 8), layout="constrained")
    for index, name in enumerate(names):
        y0 = border + (len(names) - 1 - index) * tile
        x0 = border + level * tile
        heights = grid[y0 : y0 + tile, x0 : x0 + tile]
        xy = np.arange(tile) * terrain.horizontal_scale - 4
        xx, yy = np.meshgrid(xy, xy)
        ax = fig.add_subplot(2, 3, index + 1, projection="3d")
        ax.plot_surface(xx, yy, heights, cmap="terrain", rstride=3, cstride=3)
        ax.set(
            title=name.replace("_", " "), xlabel="x (m)", ylabel="y (m)", zlabel="z (m)"
        )
        ax.set_box_aspect((8, 8, 2))
        ax.zaxis.set_major_locator(MaxNLocator(3))
        ax.scatter(0, 0, heights[tile // 2, tile // 2], color="red", s=25)
        ax.view_init(elev=30, azim=-55)
    fig.suptitle(
        "M1 UniLab | seed 17 | level 2 of 0-4 | 8 x 8 m tiles\nRed = start; move outwards. Each panel has its own height scale."
    )
    output = PACKAGE_ROOT / "terrain_preview.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
