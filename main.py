from dp_tester.results_collector import dp_results_from_sql_query
from dp_tester.query_executors import SqlAlchemyQueryExecutor
from dp_tester.dp_rewriters import PyqrlewDpRewriter
from dp_tester.table_renamers import PyqrlewTableRenamer
from dp_tester.constants import D_0, D_1

from dp_tester.generate_datasets import N_STORES
from dp_tester.partitioners import QuantityOverGroups
from dp_tester.analyzer import results_to_bucket_ids, counts_from_indexes
from dp_tester.analyzer import empirical_epsilon

from dp_tester.generate_datasets import generate_D_0_dataset, generate_adj_datasets
import json

EPSILON = 10.0
DELTA = 1e-4
COUNT_THRESHOLD = 5
RUNS = 10000
NBINS = 20
Cu = 5
QUERY = "SELECT store_id, SUM(spent) FROM transactions GROUP BY store_id"

print("Generating datasets")

generate_D_0_dataset()
generate_adj_datasets(D_1, user_id=0)

print("Generating DP results")
query_executor = SqlAlchemyQueryExecutor()
dp_rewriter = PyqrlewDpRewriter(engine=query_executor.engine, max_privacy_unit_groups=Cu)
tables = ["users", "transactions"]
table_renamer = PyqrlewTableRenamer(dp_rewriter.dataset, tables)


results = dp_results_from_sql_query(
    non_dp_query=QUERY,
    epsilon=EPSILON,
    delta=DELTA,
    runs=RUNS,
    dp_rewriter=dp_rewriter,
    query_executor=query_executor,
    table_renamer=table_renamer,
    d_0=D_0,
    adjacent_ds=[D_1],
)

filename = f"results_cu_{Cu}_eps_{EPSILON}_delta_{DELTA}.json"
with open(filename, "w") as outfile:
    json.dump(obj=results, fp=outfile)
with open(filename) as infile:
    results = json.load(infile)

print("Partitioning results and associating values to each bucket id")
groups = list(range(N_STORES))
partitioner = QuantityOverGroups(groups=groups)
partitioner.generate_buckets(results, n_bins=NBINS)
bucket_ids = results_to_bucket_ids(results, partitioner.partition_vector())

empirical_eps_per_group = {}
for i, partition_bucket_ids in enumerate(bucket_ids):
    counts_d_0 = counts_from_indexes(partition_bucket_ids[D_0], len(partitioner.buckets))
    counts_d_1 = counts_from_indexes(partition_bucket_ids[D_1], len(partitioner.buckets))
    empirical_eps_per_group[i] = empirical_epsilon(
        counts_d_0, counts_d_1, delta=DELTA, counts_threshold=COUNT_THRESHOLD, plot=False
    )
all_eps = list(empirical_eps_per_group.values())
max_eps = max(all_eps)
print(f"Epsilon used during the experiment: {EPSILON}")
print(f"Max empirical epsilon found: {max_eps}")
print(f"Did the test passed? {max_eps <= EPSILON}")
