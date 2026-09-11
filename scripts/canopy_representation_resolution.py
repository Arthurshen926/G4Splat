"""Bounded paired recipe for candidate representation resolution, not a fix.

The dense arm changes count and angular size together. It preserves the nominal
physical motion extent, not exact rendered alpha, 3D optical mass, or sampling.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RepresentationRecipe:
    rays_per_view: int
    pixel_sigma: float
    position_radius_sigmas: float

    def nominal_projected_area_ratio(self, population):
        if not isinstance(population, int) or population < 0:
            raise ValueError('Nonnegative source ray population required')
        control_count = min(population, 1024)
        if not control_count:
            return None
        return min(population, self.rays_per_view) / control_count * (self.pixel_sigma / 1.2) ** 2

    def initial_peak_opacity(self, population):
        # Match nominal low-opacity projected-area budget per seed camera.
        # Not exact alpha matching: sampling, occlusion and raster filtering
        # still differ. This is initialization only, never a training target.
        area_ratio = self.nominal_projected_area_ratio(population)
        return None if area_ratio is None else .001 / area_ratio


def representation_recipe(policy):
    if policy == 'coarse':
        return RepresentationRecipe(1024, 1.2, 8.)
    if policy == 'dense_fine':
        return RepresentationRecipe(4096, .6, 16.)
    raise ValueError('Explicit coarse or dense_fine paired policy required')


def validate_representation_scope(*, native_rgb, shape_cohort, kernel):
    if not native_rgb or shape_cohort is not None or kernel != 'native':
        raise ValueError('Native RGB/kernel and no stale shape cohort required in BOTH arms')
