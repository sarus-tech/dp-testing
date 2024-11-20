from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import create_engine, text
from dp_tester.generate_datasets import USER, PASSWORD, HOST, PORT
from dp_tester.typing import QueryResults
import typing as t


class SqlAlchemyQueryExecutor:
    def __init__(
        self,
        user: str = USER,
        password: str = PASSWORD,
        host: str = HOST,
        port: int = PORT,
        pool_size: int = 5,
    ):
        self.engine = create_engine(
            f"postgresql+psycopg2://{user}:{password}@{host}:{port}/postgres",
            pool_size=pool_size,
        )

    def run_query(self, name: str, query: str) -> t.Tuple[str, QueryResults]:
        with self.engine.begin() as conn:
            result = conn.execute(text(query)).fetchall()
            casted_result = [tuple(row) for row in result]
        return (name, casted_result)

    def run_queries(
        self, named_queries: t.Sequence[t.Tuple[str, str]]
    ) -> t.Mapping[str, QueryResults]:
        results = {}

        with ThreadPoolExecutor() as executor:
            # Submit each query as a separate task
            futures = {
                executor.submit(self.run_query, name, query): name
                for (name, query) in named_queries
            }

            # Collect results as each query completes
            for future in as_completed(futures):
                name = futures[future]
                try:
                    name, result = future.result()
                    results[name] = result
                except Exception as e:
                    print(f"Error in the query with name {name}: {e}")

        self.engine.dispose()
        return results
