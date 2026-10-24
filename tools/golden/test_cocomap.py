"""Unit tests of cocomap.py against cases worked out by hand. Run: pytest tools/golden"""

import pytest

from cocomap import evaluate

BOX = (0, 0, 10, 10)


def test_ranked_fp_between_two_tp():
    # Scores .9 TP, .8 FP, .5 TP: precision 1 up to recall .5 (51 points), then 2/3 (50 points).
    gt = {1: [(1, *BOX, 0)], 2: [(1, *BOX, 0)]}
    dets = [(1, 1, 0.9, *BOX), (2, 1, 0.8, 20, 20, 10, 10), (2, 1, 0.5, *BOX)]
    expected = (51 * 1.0 + 50 * 2 / 3) / 101
    res = evaluate(gt, dets, [1])
    assert res["map"] == pytest.approx(expected)
    assert res["ap50"] == pytest.approx(expected)


def test_iou_threshold_sweep():
    # Shifted by 1 px: IoU 90/110 = .818 passes thresholds .50-.80 (7 of 10) and fails .85-.95.
    gt = {1: [(1, *BOX, 0)]}
    res = evaluate(gt, [(1, 1, 0.9, 1, 0, 10, 10)], [1])
    assert res["map"] == pytest.approx(0.7)
    assert res["ap50"] == pytest.approx(1.0)
    assert res["ap75"] == pytest.approx(1.0)


def test_class_without_detections_averages_in_as_zero():
    gt = {1: [(1, *BOX, 0), (2, 50, 50, 10, 10, 0)]}
    res = evaluate(gt, [(1, 1, 0.9, *BOX)], [1, 2])
    assert res["map"] == pytest.approx(0.5)


def test_class_without_ground_truth_is_left_out():
    res = evaluate({1: [(1, *BOX, 0)]}, [(1, 1, 0.9, *BOX)], [1, 2])
    assert res["map"] == pytest.approx(1.0)


def test_crowd_match_is_ignored_not_a_false_positive():
    gt = {1: [(1, *BOX, 0), (1, 100, 100, 50, 50, 1)]}
    dets = [(1, 1, 0.9, *BOX), (1, 1, 0.8, 110, 110, 10, 10)]
    assert evaluate(gt, dets, [1])["map"] == pytest.approx(1.0)
