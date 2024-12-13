from dp_tester.typing import PartitionVector, OverallResults
import numpy as np
import typing as t
import numpy.typing as npt
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = False
plt.rcParams["font.size"] = 12


def results_to_bucket_ids(
    results: OverallResults, partition_vector: PartitionVector
) -> t.List[t.Dict[str, t.List[t.Union[int, None]]]]:
    ds_bucket_ids_per_partition = []
    for partitioner in partition_vector:
        bucket_ids: t.Dict[str, t.List[t.Union[int, None]]] = {}

        for ds_name, runs in results.items():
            ds_bucket_ids = []

            for query_results in runs:
                ds_bucket_ids.append(partitioner(query_results))

            bucket_ids[ds_name] = ds_bucket_ids

        ds_bucket_ids_per_partition.append(bucket_ids)

    return ds_bucket_ids_per_partition


def counts_from_indexes(
    indexes: t.List[t.Union[int, None]], max_bucket_length: int
) -> npt.NDArray[np.intp]:
    """It generates counts from a list of integers. Nones are considered
    as the max_bucket_length + 1 element of the buckets.
    """
    arr = [i if i is not None else max_bucket_length + 1 for i in indexes]
    return np.bincount(arr, minlength=max_bucket_length + 1)


def empirical_epsilon(
    counts_d_0: npt.NDArray[np.intp],
    counts_d_1: npt.NDArray[np.intp],
    delta: float,
    counts_threshold: int,
    plot: bool = False,
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

    false_positive = c_minus_cumsum_c0 / C
    false_negative = csm_c1 / C

    # plot false_positive and false_negative

    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.log((1 - delta - false_positive) / false_negative)
        b = np.log((1 - delta - false_negative) / false_positive)
        max_eps = np.nanmax([a, b])

    # plot probabilities
    if plot:
        fig, axs = plt.subplots(2, 1, sharex=True)
        ax1 = axs[0]
        ax1.plot(ratios[sorting_index])
        ax1.set_ylabel("Sorted probability ratio")
        ax1.legend()
        ax1.grid(True)

        ax2 = axs[1]
        ax2.plot(false_positive, label="FP")
        ax2.plot(false_negative, label="FN")
        ax2.set_xlabel("S_n")
        ax2.set_ylabel("Probability")
        ax2.legend()
        ax2.grid(True)

        plt.show()

    return max_eps
