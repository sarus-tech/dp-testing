from pyqrlew.wrappers import Dataset, Relation, RelationWithDpEvent
from sqlalchemy import Engine
import typing as t


class PyqrlewDpRewriter:
    def __init__(
        self,
        engine: Engine,
        dataset_name: str = "retail",
        schema: t.Optional[str] = None,
        max_privacy_unit_groups: int = 5,
    ):
        tmp_dataset = Dataset.from_database(dataset_name, engine, schema, ranges=True)
        self.dataset = tmp_dataset.users.id.with_unique_constraint()  # type: ignore
        self.max_privacy_unit_groups = max_privacy_unit_groups

    def relation_with_dpevent(
        self, query: str, epsilon: float, delta: float
    ) -> RelationWithDpEvent:
        """
        1) It parses the self.query into a qrlew relation.
        2) It defines the privacy unit
        3) It rewrites the query in DP
        4) It returns the rewritten DP query and the DP relation.
        """
        rel = Relation.from_query(query, self.dataset)
        privacy_unit = [
            ("users", [], "id"),
            ("transactions", [("user_id", "users", "id")], "id"),
        ]
        epsilon_delta = {"epsilon": epsilon, "delta": delta}
        return rel.rewrite_with_differential_privacy(
            dataset=self.dataset,
            synthetic_data=None,
            privacy_unit=privacy_unit,
            epsilon_delta=epsilon_delta,
            max_privacy_unit_groups=self.max_privacy_unit_groups,
        )

    # Utils used for debugging

    def plot_relation(self, query: str, epsilon: float, delta: float):
        from pyqrlew import utils

        rel = Relation.from_query(query, self.dataset)
        privacy_unit = [
            ("users", [], "id"),
            ("transactions", [("user_id", "users", "id")], "id"),
        ]
        epsilon_delta = {"epsilon": epsilon, "delta": delta}
        dp_rel_event = rel.rewrite_with_differential_privacy(
            dataset=self.dataset,
            synthetic_data=None,
            privacy_unit=privacy_unit,
            epsilon_delta=epsilon_delta,
            max_privacy_unit_groups=self.max_privacy_unit_groups,
        )
        dp_rel = dp_rel_event.relation()
        dp_query = dp_rel.to_query()
        print(dp_query)
        utils.display_graph(dp_rel.dot())

    def rewrite_with_differential_privacy(
        self, query: str, epsilon: float, delta: float
    ) -> str:
        dp_rel = self.relation_with_dpevent(query, epsilon, delta).relation()
        dp_query = dp_rel.to_query()
        return dp_query
