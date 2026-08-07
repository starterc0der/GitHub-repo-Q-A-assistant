from __future__ import annotations

import math

from src.trace import project_3d, vector_preview


def test_vector_preview_reports_dim_norm_and_truncated_values() -> None:
    preview = vector_preview([3.0, 4.0], n=8)

    assert preview.dim == 2
    assert preview.norm == 5.0
    assert preview.preview == [3.0, 4.0]


def test_vector_preview_truncates_to_n_values() -> None:
    preview = vector_preview([1.0] * 20, n=8)

    assert len(preview.preview) == 8
    assert preview.dim == 20


def test_project_3d_returns_origin_for_fewer_than_two_vectors() -> None:
    assert project_3d([]) == []
    assert project_3d([[1.0, 2.0, 3.0]]) == [(0.0, 0.0, 0.0)]


def test_project_3d_preserves_input_order_and_count() -> None:
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]

    projected = project_3d(vectors)

    assert len(projected) == len(vectors)
    assert all(len(p) == 3 for p in projected)


def test_project_3d_places_similar_vectors_closer_than_dissimilar_ones() -> None:
    # Two tight clusters, far apart in the original space.
    cluster_a = [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]]
    cluster_b = [[10.0, 10.0, 10.0], [10.01, 10.0, 10.0], [10.0, 10.01, 10.0]]

    projected = project_3d(cluster_a + cluster_b)
    a_points, b_points = projected[:3], projected[3:]

    def dist(p, q):
        return math.dist(p, q)

    within_a = max(dist(a_points[i], a_points[j]) for i in range(3) for j in range(3))
    across = min(dist(p, q) for p in a_points for q in b_points)

    assert across > within_a
