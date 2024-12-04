from pyqrlew.wrappers import Dataset, Relation
import typing as t


class PyqrlewTableRenamer:
    def __init__(self, pyqrlew_dataset: Dataset, tables: t.Sequence[str]):
        self.dataset = pyqrlew_dataset
        self.tables = tables

    def query_with_schema(self, query: str, schema_name: str) -> str:
        """It composes the query to add the schema to tables names."""

        new_ds = self.dataset.from_queries(
            [
                ((path[0], schema_name, path[-1]), rel.to_query())
                for (path, rel) in self.dataset.relations()
                if path[-1] in self.tables
            ]
        )
        composing_relations = [
            (
                (table,),
                Relation.from_query(f'SELECT * FROM "{schema_name}"."{table}"', new_ds),
            )
            for table in self.tables
        ]
        dp_relation = Relation.from_query(query, self.dataset)
        composed = dp_relation.compose(composing_relations)
        return composed.to_query()
