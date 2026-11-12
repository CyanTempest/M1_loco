"""Modest starting envelopes, not a guarantee of terrain success."""
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg


def terrain_cfg(family="mixed"):
    terrains = {
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20),
        # Robots spawn on the center platform and evaluation drives outward.
        # Inverted pyramids therefore climb outward; regular pyramids descend outward.
        "slope_up": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(0.0, 0.35), platform_width=2.0, border_width=0.25),
        "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(0.0, 0.35), platform_width=2.0, border_width=0.25),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20, noise_range=(0.005, 0.04), noise_step=0.005, border_width=0.25),
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20, step_height_range=(0.03, 0.16), step_width=0.30,
            platform_width=2.0, border_width=1.0, holes=False),
        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20, step_height_range=(0.03, 0.16), step_width=0.30,
            platform_width=2.0, border_width=1.0, holes=False),
    }
    if family != "mixed":
        if family not in terrains:
            raise ValueError(f"Unknown terrain family: {family}")
        terrains = {family: terrains[family]}
        terrains[family].proportion = 1.0
    return TerrainGeneratorCfg(
        seed=42, size=(8.0, 8.0), border_width=20.0,
        num_rows=10, num_cols=20 if family == "mixed" else 4,
        horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
        curriculum=True, use_cache=False, sub_terrains=terrains,
    )
