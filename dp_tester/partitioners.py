from dp_tester.typing import OverallResults, PartitionVector, Partition, QueryResults
import numpy as np
import typing as t
import datetime


class QuantityOverGroups:
    """Experiment to partition results from queries like.
    SELECT <some_group>, AGG(<some_quantity>) FROM transactions [JOIN users] GROUP BY some_group

    The output space S is partitioned as all possible pairs group_id and quantity interval
    plus the empty set.
    """

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

    def __init__(self, groups=t.List[QuantityOverGroupsGroupType]):
        self.groups = groups
        self.buckets: t.List[t.Optional[t.Tuple[float, float]]]

    def partition(self, group: QuantityOverGroupsGroupType) -> Partition:
        def partition_function(query_results: QueryResults) -> t.Union[int, None]:
            query_results_as_dict = dict(query_results)
            groups_in_result = query_results_as_dict.keys()
            if group not in groups_in_result:
                return None
            else:
                buckets_without_empty = filter(None, self.buckets)
                for index, bucket in enumerate(buckets_without_empty):
                    (min_value, max_value) = bucket
                    quantity = query_results_as_dict.get(group)
                    assert quantity is not None
                    if max_value > quantity >= min_value:
                        return index
                raise ValueError(
                    f"I Couldn't associated quantity {quantity} to any bucket id."
                    " Please make sure buckets are right."
                )

        return partition_function

    def partition_vector(self) -> PartitionVector:
        partitions = []
        for group in self.groups:
            partitions.append(self.partition(group))
        return partitions

    def generate_buckets(self, results: OverallResults, n_bins: int):
        """It generates buckets from the results. It concatenates
        all `some_quantity` from all results and uses np.histogram
        to find bin edges.
        """
        X: t.List[t.List[float]] = []
        for _, res in results.items():
            for query_res in res:
                quantity = [t.cast(float, r[-1]) for r in query_res]
                X.append(quantity)
        concatenated = np.concatenate(X, dtype="float")
        _, bin_edges = np.histogram(
            concatenated,
            bins=n_bins,
            range=(np.min(concatenated), np.max(concatenated) + 1),
        )
        self.buckets = list(zip(bin_edges[:-1], bin_edges[1:]))
