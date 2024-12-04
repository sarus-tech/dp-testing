from dp_tester.typing import (
    OverallResults,
)
import numpy as np
import typing as t
import datetime

QuantityOverGroupsGroupType = t.Union[
    int,
    float,
    str,
    datetime.date,
    datetime.time,
    datetime.datetime,
    datetime.timedelta,
]
QuantityOverGroupsQuantityType = t.List[t.Union[int, float]]
Row = t.Tuple[QuantityOverGroupsGroupType, QuantityOverGroupsQuantityType]
QueryResults = t.Sequence[Row]


class QuantityOverGroups:
    """Experiment to partition results from queries like.
    SELECT <some_group>, <some_quantity> FROM transactions [JOIN users] GROUP BY some_group

    The final partitions will look like:
    dataset-group: [some_quantity, ...]

    all possible groups should be considered, e.g. all possible store_id if
    grouping by store_id.
    """

    def __init__(self, groups=t.List[QuantityOverGroupsGroupType]):
        self.groups = groups
        self.buckets: t.List[
            t.Union[t.Tuple[QuantityOverGroupsGroupType, t.Tuple[float, float]]]
        ]

    def generate_buckets(self, results: OverallResults, n_float_buckets: int):
        """It generates len(self.groups) * n_float_buckets buckets + 1:
        Each bucket is related to a specific (group, (x_min, x_max) )
        the + 1 is needed to indicate that there are no results.
        """
        X: t.List[t.List[float]] = []
        for _, res in results.items():
            for query_res in res:
                quantity = [float(r[-1]) for r in query_res]
                X.append(quantity)
        X = np.concatenate(X, dtype="float")
        _, bin_edges = np.histogram(
            X, bins=n_float_buckets, range=(np.min(X), np.max(X) + 1)
        )
        float_buckets = list(zip(bin_edges[:-1], bin_edges[1:]))
        groups = [group for group in self.groups for _ in range(len(float_buckets))]
        self.buckets = list(zip(groups, float_buckets * len(self.groups))) + [None]

    def bucket_id(self, query_results: QueryResults) -> t.List[int]:
        """Function used by `partition_results_to_bucket_ids` for getting
        bucket index from query results."""

        assert self.buckets is not None, "Buckets are not defined"
        no_group_present_idx = len(self.buckets) - 1

        if len(query_results) == 0:
            return [no_group_present_idx]

        query_results_as_dict = dict(query_results)
        groups_in_result = query_results_as_dict.keys()
        bucket_indexes = []

        for index, bucket in enumerate(self.buckets[:-1]):
            group_id, (min_value, max_value) = bucket
            if group_id not in groups_in_result:
                bucket_indexes.append(no_group_present_idx)
            else:
                quantity = query_results_as_dict.get(group_id)
                if max_value > quantity >= min_value:
                    bucket_indexes.append(index)
                else:
                    bucket_indexes.append(no_group_present_idx)

        bucket_indexes.append(no_group_present_idx)
        return bucket_indexes
