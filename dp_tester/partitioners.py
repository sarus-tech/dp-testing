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
QuantityOverGroupsQuantityType = t.Union[int, float]
Row = t.Tuple[QuantityOverGroupsGroupType, QuantityOverGroupsQuantityType]
QueryResults = t.Sequence[Row]


class QuantityOverGroups:
    """Experiment to partition results from queries like.
    SELECT <some_group>, AGG(<some_quantity>) FROM transactions [JOIN users] GROUP BY some_group

    The output space S is partitioned as all possible pairs group_id and quantity interval
    plus the empty set.
    """

    def __init__(self, groups=t.List[QuantityOverGroupsGroupType]):
        self.groups = groups
        self.buckets: t.List[
            t.Optional[t.Union[t.Tuple[QuantityOverGroupsGroupType, t.Tuple[float, float]]]]
        ]

    def generate_buckets(self, results: OverallResults, n_float_buckets: int):
        """It generates len(self.groups) * n_float_buckets buckets + 1:
        Each bucket is related to a specific (group, (x_min, x_max) )
        the + 1 is needed to indicate that there are no results.

         example of buckets 
        [
            (0, (0., 5.)),
            (0, (5., 10.)),
            (1, (0., 5.)),
            (1, (5., 10.)),
            ...
            None,
        ]
        """
        X: t.List[t.List[float]] = []
        for _, res in results.items():
            for query_res in res:
                quantity = [t.cast(float, r[-1]) for r in query_res]
                X.append(quantity)
        concatenated = np.concatenate(X, dtype="float")
        _, bin_edges = np.histogram(
            concatenated, bins=n_float_buckets, range=(np.min(concatenated), np.max(concatenated) + 1)
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
            return [no_group_present_idx] * len(self.buckets)

        query_results_as_dict = dict(query_results)
        groups_in_result = query_results_as_dict.keys()
        bucket_indexes = []

        buckets_without_empty = filter(None, self.buckets)
        for index, bucket in enumerate(buckets_without_empty):
            group_id, (min_value, max_value) = bucket
            if group_id not in groups_in_result:
                bucket_indexes.append(no_group_present_idx)
            else:
                quantity = query_results_as_dict.get(group_id)
                assert quantity is not None
                if max_value > quantity >= min_value:
                    bucket_indexes.append(index)
                else:
                    bucket_indexes.append(no_group_present_idx)

        bucket_indexes.append(no_group_present_idx)
        return bucket_indexes
