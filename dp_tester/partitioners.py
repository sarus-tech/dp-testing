from dp_tester.typing import (
    OverallResults,
    PartitionedResults,
)
import numpy as np
import typing as t


class QuantityOverGroups:
    """Experiment to partition results from queries like.
    SELECT <some_group>, <some_quantity> FROM transactions [JOIN users] GROUP BY some_group

    Each partition will have
    """

    def __init__(self, groups=t.List[t.Any]):
        self.groups = groups
        self.buckets: t.List[t.Union[t.Tuple[float, float], None]]

    def partition_results(self, results: OverallResults) -> PartitionedResults:
        res: PartitionedResults = {}

        for ds_name, runs in results.items():
            runs_res: PartitionedResults = {group: [] for group in self.groups}
            for run_results in runs:
                res_as_dict = dict(run_results)

                for group in self.groups:
                    runs_res[group].append(res_as_dict.get(group, None))

            for group, partition_results in runs_res.items():
                res[f"{ds_name}-{group}"] = partition_results

        return res

    # TODO: improve code readability of partition_results.
    # def partition_runs(self, run_results: QueryResults):

    def generate_buckets(self, partitioned_results: PartitionedResults, nbuckets: int):
        """concatenate all values from all the partitioned results and use
        numpy.histogram to generate the bin edges.

        It returns a list with tuples (lower_edge, higher_edge)
        """
        X = []
        for _, res in partitioned_results.items():
            X.append(res)
        X = np.concatenate(np.array(X, dtype="float"))
        _, bin_edges = np.histogram(
            X, bins=nbuckets, range=(np.nanmin(X), np.nanmax(X) + 1)
        )
        self.buckets = list(zip(bin_edges[:-1], bin_edges[1:])) + [None]

    def bucket_id(self, data_point: t.Union[float, None]) -> int:
        assert self.buckets is not None, "Buckets are not defined"
        if data_point is None:
            return len(self.buckets) - 1
        for i, bucket_value in enumerate(self.buckets[:-1]):
            assert bucket_value is not None
            (min_value, max_value) = bucket_value
            if max_value > data_point >= min_value:
                return i
        raise ValueError("Couldn't allocate value to a bucket")
