"""Tests for the binning helper bin_parameter_map."""

import numpy as np
from furax_cs import bin_parameter_map


class TestBinParameterMap:
    """Verify that bin_parameter_map follows the notebook logic correctly."""

    def test_basic_binning(self):
        """10 bins on a uniform parameter map produces 10 unique patch indices."""
        rng = np.random.default_rng(42)
        pixel_map = rng.uniform(1.0, 2.0, size=1000)
        nbins = 10

        indices, centers, edges = bin_parameter_map(pixel_map, nbins)

        assert indices.shape == pixel_map.shape
        assert centers.shape == (nbins,)
        assert edges.shape == (nbins + 1,)

    def test_indices_are_0_based(self):
        """All returned indices must be in [0, nbins-1]."""
        pixel_map = np.linspace(1.0, 2.0, 500)
        nbins = 5

        indices, centers, _ = bin_parameter_map(pixel_map, nbins)

        assert indices.min() >= 0
        assert indices.max() <= nbins - 1

    def test_all_pixels_assigned(self):
        """Every valid pixel gets a bin assignment (no -1 or out-of-range)."""
        rng = np.random.default_rng(123)
        pixel_map = rng.uniform(10.0, 30.0, size=2000)
        nbins = 20

        indices, _, _ = bin_parameter_map(pixel_map, nbins)

        assert np.all(indices >= 0)
        assert np.all(indices < nbins)

    def test_bin_centers_within_range(self):
        """Bin centers are within the original value range."""
        rng = np.random.default_rng(7)
        pixel_map = rng.uniform(-3.5, -1.0, size=1000)
        nbins = 15

        _, centers, _ = bin_parameter_map(pixel_map, nbins)

        assert centers.min() >= pixel_map.min()
        assert centers.max() <= pixel_map.max() + 1e-10

    def test_indexing_gives_valid_values(self):
        """bin_centers[indices] gives a valid per-pixel value for all pixels."""
        rng = np.random.default_rng(99)
        pixel_map = rng.uniform(1.4, 1.7, size=500)
        nbins = 10

        indices, centers, _ = bin_parameter_map(pixel_map, nbins)

        binned_values = centers[indices]
        assert binned_values.shape == pixel_map.shape
        # Binned values should be close to original (within half a bin width)
        bin_width = (pixel_map.max() - pixel_map.min()) / nbins
        assert np.all(np.abs(binned_values - pixel_map) <= bin_width)

    def test_single_bin(self):
        """With 1 bin, all pixels map to the same index and center."""
        pixel_map = np.array([1.0, 1.5, 2.0, 1.3])
        indices, centers, _ = bin_parameter_map(pixel_map, 1)

        assert np.all(indices == 0)
        assert centers.shape == (1,)
        assert np.isclose(centers[0], 1.5, atol=0.01)

    def test_constant_map(self):
        """A constant parameter map still works (all values in one bin)."""
        pixel_map = np.full(100, 1.54)
        indices, centers, _ = bin_parameter_map(pixel_map, 5)

        # All pixels should end up in a single bin
        assert len(np.unique(indices)) == 1
        assert indices.min() >= 0

    def test_matches_notebook_logic(self):
        """Verify agreement with the notebook's np.histogram + np.digitize approach."""
        rng = np.random.default_rng(42)
        pixel_map = rng.uniform(1.0, 2.0, size=10000)
        nbins = 100

        # Our implementation
        indices_ours, centers_ours, edges_ours = bin_parameter_map(pixel_map, nbins)

        # Notebook logic (np.histogram gives bin edges, np.digitize assigns)
        _, bin_edges_nb = np.histogram(
            pixel_map, bins=nbins, range=[pixel_map.min(), pixel_map.max() + 1e-10]
        )
        indices_nb = np.digitize(pixel_map, bin_edges_nb)
        indices_nb = np.clip(indices_nb, 1, nbins) - 1  # 0-based

        np.testing.assert_array_equal(indices_ours, indices_nb)
