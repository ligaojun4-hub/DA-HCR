"""Meaningful geometry and known-answer metric checks; no torch required."""

import unittest

import numpy as np
from PIL import Image

from revision_protocol import (
    PERSPECTIVE_DIRECTIONS, PERSPECTIVE_RHOS, SPECS, SPECS_BY_NAME,
    apply_new_pil, bit_flip_diagnostics, bit_flip_rate, code_diagnostics,
    evaluate_codes, hamming_distance_matrix, perspective_homography,
    perspective_source_points, perspective_target_points, project_points,
    sampling_boundary_report, warp_perspective,
)


class PerspectiveGeometryTests(unittest.TestCase):
    def test_all_corner_correspondences_and_non_affine(self):
        for width, height in ((224, 224), (128, 96)):
            for rho in PERSPECTIVE_RHOS:
                for direction in PERSPECTIVE_DIRECTIONS:
                    with self.subTest(width=width, rho=rho, direction=direction):
                        matrix = perspective_homography(width, height, rho, direction)
                        actual = project_points(perspective_source_points(width, height, rho, direction), matrix)
                        np.testing.assert_allclose(actual, perspective_target_points(width, height), atol=1e-9)
                        self.assertGreater(np.linalg.norm(matrix[2, :2]), 0)

    def test_identity_has_no_interpolation_change(self):
        pixels = np.random.default_rng(7).integers(0, 256, (96, 128, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        for direction in PERSPECTIVE_DIRECTIONS:
            np.testing.assert_array_equal(perspective_homography(128, 96, 0, direction), np.eye(3))
            np.testing.assert_array_equal(np.asarray(warp_perspective(image, 0, direction)), pixels)

    def test_full_sampling_grid_stays_in_input_and_has_no_fill(self):
        pixels = np.broadcast_to(np.array((121, 131, 141), dtype=np.uint8), (96, 128, 3)).copy()
        image = Image.fromarray(pixels)
        for rho in PERSPECTIVE_RHOS:
            for direction in PERSPECTIVE_DIRECTIONS:
                with self.subTest(rho=rho, direction=direction):
                    matrix = perspective_homography(128, 96, rho, direction)
                    self.assertEqual(sampling_boundary_report(matrix, 128, 96)["num_outside"], 0)
                    np.testing.assert_array_equal(np.asarray(warp_perspective(image, rho, direction)), pixels)

    def test_coordinate_grid_obeys_independent_top_inverse_formula(self):
        width = height = 224
        yy, xx = np.mgrid[0:height, 0:width]
        pixels = np.stack((xx, yy, np.full_like(xx, 173)), axis=-1).astype(np.uint8)
        rho = 0.20
        warped = np.asarray(warp_perspective(Image.fromarray(pixels), rho, "top"))
        for x, y in ((0, 0), (223, 0), (0, 223), (223, 223), (47, 63), (150, 100), (111, 111)):
            u, v = x / (width-1), y / (height-1)
            denominator = 1-2*rho*v
            expected_x = ((1-2*rho)*u + rho*(1-v)) / denominator * (width-1)
            expected_y = (1-2*rho)*v / denominator * (height-1)
            self.assertLessEqual(abs(float(warped[y, x, 0])-expected_x), 1.01)
            self.assertLessEqual(abs(float(warped[y, x, 1])-expected_y), 1.01)
            self.assertEqual(warped[y, x, 2], 173)

    def test_strength_moves_top_edges_monotonically(self):
        offsets = [perspective_source_points(224, 224, rho, "top")[0, 0] for rho in PERSPECTIVE_RHOS]
        self.assertTrue(all(left < right for left, right in zip(offsets, offsets[1:])))
        self.assertEqual(len({tuple(perspective_source_points(224, 224, .2, d).ravel()) for d in PERSPECTIVE_DIRECTIONS}), 4)

    def test_invalid_or_degenerate_geometry_rejected(self):
        for rho in (-.01, .5, float("nan")):
            with self.assertRaises(ValueError):
                perspective_homography(224, 224, rho, "top")
        with self.assertRaises(ValueError):
            perspective_homography(1, 224, .1, "top")

    def test_new_identity_matches_declared_working_resize(self):
        pixels = np.random.default_rng(8).integers(0, 256, (67, 103, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        actual = apply_new_pil(image, {"kind": "perspective", "rho": 0, "direction": "top"}, 224)
        expected = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_legacy_routes_are_not_silently_reimplemented(self):
        for name in ("clean", "weak", "medium", "strong"):
            with self.assertRaises(ValueError):
                apply_new_pil(Image.new("RGB", (224, 224)), SPECS_BY_NAME[name])
        self.assertEqual(sum(spec["kind"] == "perspective" for spec in SPECS), 12)

    def test_brightness_has_correct_sign_and_clipping(self):
        pixels = np.broadcast_to(np.array((10, 100, 250), dtype=np.uint8), (224, 224, 3)).copy()
        image = Image.fromarray(pixels)
        plus = np.asarray(apply_new_pil(image, SPECS_BY_NAME["brightness_p30"]))
        minus = np.asarray(apply_new_pil(image, SPECS_BY_NAME["brightness_m30"]))
        np.testing.assert_array_equal(plus[0, 0], (40, 130, 255))
        np.testing.assert_array_equal(minus[0, 0], (0, 70, 220))


class RetrievalMetricTests(unittest.TestCase):
    def test_hamming_dot_formula_matches_bitwise_xor(self):
        codes = np.array(((1, 1), (1, -1), (-1, 1), (-1, -1)), dtype=np.int8)
        expected = np.count_nonzero(codes[:, None, :] != codes[None, :, :], axis=2)
        np.testing.assert_array_equal(hamming_distance_matrix(codes, codes), expected)

    def test_known_full_gallery_ap_and_chunk_invariance(self):
        queries = np.array(((1, 1), (1, -1)))
        gallery = np.array(((1, 1), (1, -1), (-1, 1)))
        kwargs = dict(query_codes=queries, query_labels=[1, 0], gallery_codes=gallery,
                      gallery_labels=[1, 0, 1], query_ids=["q1", "q2"], gallery_ids=["g1", "g2", "g3"])
        first = evaluate_codes(**kwargs, chunk_size=1)
        second = evaluate_codes(**kwargs, chunk_size=32)
        np.testing.assert_allclose(first["per_query_ap"], (5/6, 1), atol=1e-12)
        self.assertAlmostEqual(first["mAP"], 11/12)
        np.testing.assert_array_equal(first["per_query_ap"], second["per_query_ap"])

    def test_gallery_id_tie_rule_is_invariant_to_row_permutation(self):
        gallery = np.ones((3, 2))
        first = evaluate_codes([[1, 1]], [1], gallery, [0, 1, 1], ["q"], ["b", "a", "c"])
        second = evaluate_codes([[1, 1]], [1], gallery, [1, 1, 0], ["q"], ["c", "a", "b"])
        self.assertAlmostEqual(first["mAP"], 5/6)
        self.assertEqual(first["mAP"], second["mAP"])
        self.assertEqual(first["mean_tied_gallery_fraction"], 1)

    def test_self_exclusion_changes_rank_and_denominator_correctly(self):
        result = evaluate_codes([[1, 1]], [1], np.ones((3, 2)), [1, 0, 1], ["a"], ["a", "b", "c"], exclude_self=True)
        self.assertAlmostEqual(result["mAP"], .5)
        self.assertEqual(result["relevant_counts"][0], 1)
        self.assertEqual(result["available_gallery_counts"][0], 2)

    def test_no_relevant_or_empty_gallery_is_explicitly_accounted(self):
        result = evaluate_codes([[1, 1], [1, 1]], [1, 2], [[1, 1]], [1], ["q1", "q2"], ["g"])
        self.assertEqual(result["mAP"], 1)
        self.assertEqual(result["num_query_total"], 2)
        self.assertEqual(result["num_query_used"], 1)
        self.assertEqual(result["num_query_without_relevant"], 1)
        self.assertTrue(np.isnan(result["per_query_ap"][1]))
        empty = evaluate_codes([[1, 1]], [1], np.empty((0, 2)), [], ["q"], [])
        self.assertIsNone(empty["mAP"])
        self.assertEqual(empty["num_query_without_gallery"], 1)

    def test_zero_codes_and_duplicate_gallery_ids_rejected(self):
        with self.assertRaises(ValueError):
            hamming_distance_matrix([[1, 0]], [[1, 1]])
        with self.assertRaises(ValueError):
            evaluate_codes([[1, 1]], [1], np.ones((2, 2)), [1, 1], ["q"], ["g", "g"])

    def test_bfr_redundancy_and_query_alignment(self):
        clean = np.array(((1, 1), (-1, -1)))
        degraded = np.array(((1, -1), (-1, -1)))
        result = bit_flip_diagnostics(clean, degraded, ["a", "b"], ["a", "b"])
        self.assertEqual(bit_flip_rate(clean, degraded), .25)
        self.assertEqual(result["mean_hamming_distance"], .5)
        self.assertEqual(result["sign_consistency"], .75)
        np.testing.assert_array_equal(result["per_query_hamming"], (1, 0))
        with self.assertRaises(ValueError):
            bit_flip_diagnostics(clean, degraded, ["a", "b"], ["b", "a"])

    def test_constant_codes_are_diagnosed_as_collapsed(self):
        collapsed = code_diagnostics(np.ones((4, 2)))
        balanced = code_diagnostics([[1, 1], [-1, -1]])
        self.assertEqual(collapsed["num_unique_codes"], 1)
        self.assertEqual(collapsed["num_constant_bits"], 2)
        self.assertEqual(collapsed["mean_bit_entropy_bits"], 0)
        self.assertEqual(balanced["num_constant_bits"], 0)
        self.assertEqual(balanced["mean_bit_entropy_bits"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
