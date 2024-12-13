import typing as t
from collections.abc import Callable


DataPoint = t.TypeVar("DataPoint")
Row = t.Tuple[DataPoint, ...]
QueryResults = t.Sequence[Row]
OverallResults = t.Mapping[str, t.Sequence[QueryResults]]

Partition = Callable[[QueryResults], t.Union[int, None]]
PartitionVector = t.Sequence[Partition]


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
    """A protocol for modifying SQL queries to reference a specific database schema.
    It is assumed that D_0 and D_1 have the same tables name but they are under
    2 different schema, in the default one the D_0 and in the `schema_name` D_1.
    """

    def query_with_schema(self, query: str, schema_name: str) -> str: ...


class Partitioner(t.Protocol):
    """A protocol that defines methods for partitioning results and assigning data points to buckets."""

    def partition_vector(self, query_results: QueryResults) -> PartitionVector: ...

    """It returns a list of partition function used to map a QueryResults to a single
    bucket id.
    """
