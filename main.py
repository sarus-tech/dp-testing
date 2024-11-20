from dp_tester.generate_dp_results import generate_dp_results
from dp_tester.query_executors import SqlAlchemyQueryExecutor
from dp_tester.dp_rewriters import PyqrlewDpRewriter
from dp_tester.table_renamers import PyqrlewTableRenamer
from dp_tester.generate_datasets import D_0

from dp_tester.generate_datasets import N_STORES
from dp_tester.partitioners import QuantityOverGroups
from dp_tester.analyzer import partition_results_to_bucket_ids, counts_from_indexes
from dp_tester.analyzer import empirical_epsilon

from dp_tester.generate_datasets import generate_D_0_dataset, generate_adj_datasets, D_1
import json

EPSILON = 1.0
DELTA = 1e-4
COUNT_THRESHOLD = 5
RUNS = 100
NBINS = 20
# QUERY = "SELECT store_id, SUM(spent) FROM transactions GROUP BY store_id"
QUERY = "SELECT store_id, SUM(spent) FROM transactions GROUP BY store_id"

# print("Generating datasets")
# generate_D_0_dataset()
# generate_adj_datasets(D_1, user_id=0)


print("Generating DP results")
query_executor = SqlAlchemyQueryExecutor()
dp_rewriter = PyqrlewDpRewriter(engine=query_executor.engine)
table_renamer = PyqrlewTableRenamer(dp_rewriter.dataset)

results = generate_dp_results(
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

with open("results.json", "w") as outfile:
    json.dump(obj=results, fp=outfile)
with open("results.json") as infile:
    results = json.load(infile)

print("Partitioning results and associating values to each bucket id")
partitioner = QuantityOverGroups(groups=range(N_STORES))
partitioned_results = partitioner.partition_results(results)
partitioner.generate_buckets(partitioned_results, nbuckets=NBINS)
bucket_ids = partition_results_to_bucket_ids(partitioned_results, partitioner.bucket_id)

print("Generating Counts")
d_0_d_1_counts = {}
for group in partitioner.groups:
    buckets_ids_d_0 = bucket_ids[f"{D_0}-{group}"]
    buckets_ids_d_1 = bucket_ids[f"{D_1}-{group}"]
    counts_d_0 = counts_from_indexes(buckets_ids_d_0, NBINS)
    counts_d_1 = counts_from_indexes(buckets_ids_d_1, NBINS)
    d_0_d_1_counts[f"{D_0}-{D_1}-{group}"] = (counts_d_0, counts_d_1)

print("Computing empirical epsilons")
empirical_epsilons = {}
for name, (count_d_0, counts_d_1) in d_0_d_1_counts.items():
    empirical_epsilons[name] = empirical_epsilon(
        count_d_0, counts_d_1, delta=DELTA, counts_threshold=COUNT_THRESHOLD
    )
empirical_epsilons_values = list(empirical_epsilons.values())
max_eps = max(empirical_epsilons_values)

print(f"Epsilon used during the experiment: {EPSILON}")
print(f"Max empirical epsilon found: {max_eps}")
print(f"Did the test passed? {max_eps < EPSILON}")
