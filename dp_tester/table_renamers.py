from pyqrlew.wrappers import Dataset, Relation


class PyqrlewTableRenamer:
    def __init__(self, pyqrlew_dataset: Dataset):
        self.dataset = pyqrlew_dataset

    def query_with_schema(self, query: str, schema_name: str) -> str:
        """It composes the query to add the schema to tables names."""

        new_ds = self.dataset.from_queries(
            [
                ((path[0], schema_name, path[-1]), rel.to_query())
                for (path, rel) in self.dataset.relations()
                if "users" in path or "transactions" in path
            ]
        )

        composing_relations = [
            (
                ("users",),
                Relation.from_query(f'SELECT * FROM "{schema_name}"."users"', new_ds),
            ),
            (
                ("transactions",),
                Relation.from_query(
                    f'SELECT * FROM "{schema_name}"."transactions"', new_ds
                ),
            ),
        ]
        dp_relation = Relation.from_query(query, self.dataset)
        composed = dp_relation.compose(composing_relations)
        return composed.to_query()
