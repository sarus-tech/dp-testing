from dp_tester.typing import BucketIdFromDataPoint, PartitionedResults
import numpy as np
import typing as t
import numpy.typing as npt


def partition_results_to_bucket_ids(
    results: PartitionedResults, bucket_id_func: BucketIdFromDataPoint
) -> t.Dict[str, t.List[int]]:
    bucket_ids = {}
    for partition_key, partition_values in results.items():
        bucket_idx = [bucket_id_func(value) for value in partition_values]
        bucket_ids[partition_key] = bucket_idx
    return bucket_ids


def counts_from_indexes(
    indexes: t.List[int], max_bucket_length: int
) -> npt.NDArray[np.intp]:
    """It generates counts from a list of integers"""
    arr = np.array(indexes)
    return np.bincount(arr, minlength=max_bucket_length)


def empirical_epsilon(
    counts_d_0: npt.NDArray[np.intp],
    counts_d_1: npt.NDArray[np.intp],
    delta: float,
    counts_threshold: int,
) -> float:
    assert np.sum(counts_d_0) == np.sum(counts_d_1)
    assert np.ndim(counts_d_0) == np.ndim(counts_d_1) == 1
    C = np.sum(counts_d_0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = counts_d_0 / counts_d_1

    sorting_index = np.argsort(ratios)[::-1]

    c_minus_cumsum_c0 = C - np.cumsum(
        np.array(counts_d_0[sorting_index], dtype="float")
    )
    c_minus_cumsum_c0[np.where(c_minus_cumsum_c0 < counts_threshold)] = np.nan

    csm_c1 = np.cumsum(np.array(counts_d_1[sorting_index], dtype="float"))
    csm_c1[np.where(csm_c1 < counts_threshold)] = np.nan

    false_positive = c_minus_cumsum_c0 / C  # equivalent to (1-cumsum(c0)/C)
    false_negative = csm_c1 / C

    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.log((1 - delta - false_positive) / false_negative)
        b = np.log((1 - delta - false_negative) / false_positive)
        max_eps = np.nanmax([a, b])
    return max_eps
