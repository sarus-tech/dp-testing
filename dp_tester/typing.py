import typing as t
from collections.abc import Callable


DataPoint = t.TypeVar("DataPoint")
Row = t.Tuple[DataPoint, ...]
QueryResults = t.List[Row]
OverallResults = t.Dict[str, t.List[QueryResults]]

BucketIdFromDataPoint = Callable[[DataPoint], int]
PartitionedResults = t.Dict[str, t.List[DataPoint]]

Bucket = t.TypeVar("Bucket")
Buckets = t.List[Bucket]


class DPRewriter(t.Protocol):
    def rewrite_with_differential_privacy(
        self, query: str, epsilon: float, delta: float
    ) -> str: ...


class QueryExecutor(t.Protocol):
    def run_query(self, schema: str, query: str) -> t.Tuple[str, QueryResults]: ...
    def run_queries(
        self, named_queries: t.Sequence[t.Tuple[str, str]]
    ) -> t.Mapping[str, QueryResults]: ...


class TableRenamer(t.Protocol):
    def query_with_schema(self, query: str, schema_name: str) -> str: ...


class Partitioner(t.Protocol):
    def partition_results(self, results: OverallResults) -> PartitionedResults: ...
    def generate_buckets(
        self, partitioned_results: PartitionedResults, nbuckets: int
    ) -> None: ...
    def bucket_id(self, data_point: DataPoint) -> int: ...
