from dp_tester.typing import (
    QueryResults,
    OverallResults,
    DPRewriter,
    QueryExecutor,
    TableRenamer,
)
from tqdm import tqdm
import typing as t


def dp_results_from_sql_query(
    non_dp_query: str,
    epsilon: float,
    delta: float,
    runs: int,
    dp_rewriter: DPRewriter,
    query_executor: QueryExecutor,
    table_renamer: TableRenamer,
    d_0: str,
    adjacent_ds: t.Sequence[str],
) -> OverallResults:
    """It executes the DP query many times to the D_0
    and to all the adjacent datasets. It returns a dictionary with `run` results
    for each dataset.
    """
    dp_query = dp_rewriter.rewrite_with_differential_privacy(
        non_dp_query, epsilon, delta
    )
    results: t.Dict[str, t.List[QueryResults]] = {}
    results = {adj: [] for adj in adjacent_ds}
    results[d_0] = []

    for adj_ds_name in adjacent_ds:
        adj_query = table_renamer.query_with_schema(dp_query, adj_ds_name)
        for _ in tqdm(range(runs)):
            queries = [(d_0, dp_query), (adj_ds_name, adj_query)]
            res = query_executor.run_queries(queries)
            for schema_name, query_res in res.items():
                results[schema_name].append(query_res)
    return results
