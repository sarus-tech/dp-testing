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
    """A protocol that defines an interface for rewriting SQL queries to
    incorporate differential privacy mechanisms.
    """

    def rewrite_with_differential_privacy(
        self, query: str, epsilon: float, delta: float
    ) -> str: ...

    """Rewrites the provided SQL query into differentially private one
        with epsilon and delta as privacy parameters.
    """


class QueryExecutor(t.Protocol):
    """A protocol defining methods for executing SQL queries and retrieving their results."""
    def run_query(self, schema: str, query: str) -> t.Tuple[str, QueryResults]: ...
    """Run a single query and retrieve the results"""
    
    def run_queries(
        self, named_queries: t.Sequence[t.Tuple[str, str]]
    ) -> t.Mapping[str, QueryResults]: ...
    """Run a multiple queries in parallel and retrieve their results"""


class TableRenamer(t.Protocol):
    """A protocol for modifying SQL queries to reference a specific database schema."""
    def query_with_schema(self, query: str, schema_name: str) -> str: ...


class Partitioner(t.Protocol):
    """A protocol that defines methods for partitioning results and assigning data points to buckets."""
    def partition_results(self, results: OverallResults) -> PartitionedResults: ...

    def generate_buckets(
        self, partitioned_results: PartitionedResults, nbuckets: int
    ) -> None: ...
    def bucket_id(self, data_point: DataPoint) -> int: ...
    """Function used to associate any datapoint to a single bucket id.
    It returns the index of the bucket associated with such datapoint
    """
