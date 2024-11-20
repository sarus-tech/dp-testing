from sqlalchemy import Column, ForeignKey, MetaData, Table, create_engine, types, text
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import typing as t
import numpy.typing as npt

from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateSchema
from sqlalchemy.future import select

# For reproducibility
np.random.seed(42)

PORT: int = 5433
USER: str = "postgres"
HOST: str = "localhost"
PASSWORD: str = "pyqrlew-db"

N_USERS = 100
N_TRANSACTIONS = 10000
N_STORES = 200
N_OTHER_GROUPS = 10

D_0 = "d_0"
D_1 = "d_1"


def db_engine():
    return create_engine(
        f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/postgres"
    )


def create_ds_from_metadata(schema: t.Optional[str] = None):
    engine = db_engine()
    metadata = MetaData(schema)
    users = Table(
        "users",
        metadata,
        Column("id", types.Integer, primary_key=True),
        Column("income", types.Float, nullable=False),
    )
    transactions = Table(
        "transactions",
        metadata,
        Column("id", types.Integer, primary_key=True),
        Column(
            "user_id",
            types.Integer,
            ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        Column("store_id", types.Integer, nullable=False),
        Column("other_id", types.Integer, nullable=False),
        Column("spent", types.Float, nullable=False),
        # Column("datetime", types.DateTime(), nullable=False),
    )
    user_table = "users" if schema is None else f'{schema}."users"'
    transaction_table = "transactions" if schema is None else f'{schema}."transactions"'

    with engine.begin() as conn:
        conn.execute(
            text(f"DROP TABLE IF EXISTS {user_table}, {transaction_table} CASCADE")
        )

    metadata.create_all(engine)
    return (users, transactions)


def generate_D_0_dataset():
    engine = db_engine()
    _ = create_ds_from_metadata()
    users, trans = create_users_and_transactions_dfs()
    users.to_sql(
        name="users",
        con=engine,
        if_exists="append",
        index=False,
    )
    trans.to_sql(
        name="transactions",
        con=engine,
        if_exists="append",
        index=False,
    )


def create_users_and_transactions_dfs():
    """Table with 100 unique users.
    For our experiment the PU is the id of this table.
    """
    user_ids = np.arange(N_USERS)

    # Income normally distributed
    user_income = np.random.normal(loc=40000, scale=10000, size=100)

    users_df = pd.DataFrame({"id": user_ids, "income": user_income})
    transactions_df = create_transactions(
        user_ids, n_transactions=N_TRANSACTIONS, max_contributions=500
    )
    return users_df, transactions_df


def create_transactions(
    user_ids: npt.NDArray[np.intp],
    n_transactions: int = 10000,
    max_contributions: int = 500,
) -> pd.DataFrame:
    """It creates the transactions dataframe with n_transactions rows
    - The user_id 0 (umax) it affects max_contributions rows.
    - There are 200 different store_id and umax affects them all.
    - There are 200 different other_id and each user is associated to at most 1
    of these ids.
    """
    assert n_transactions <= max_contributions * len(user_ids)
    assert max_contributions >= N_STORES

    spent = np.random.uniform(5, 500, n_transactions)
    base_datetime = datetime(2024, 10, 15, 18, 41, 26, 332789)

    transactions_df = pd.DataFrame(
        {
            "id": np.arange(n_transactions),
            "spent": spent,
            # "datetime": [
            #     base_datetime - timedelta(days=np.random.randint(0, 365))
            #     for _ in range(n_transactions)
            # ],
        }
    )

    # User with user_id = 0 is the one with max contributions.
    # It affects all the groups in different way
    # stores with high id number will be affected more than stores with low
    # id number.
    user_with_max_contributions = 0
    user_contributions = np.zeros(len(user_ids))
    users_and_stores = []

    store_id_contribution_c = np.linspace(0, N_STORES - 1, N_STORES)
    store_id_contribution_pb = store_id_contribution_c / np.sum(store_id_contribution_c)

    umax = [user_with_max_contributions] * max_contributions
    stores_affected_by_umax = list(np.arange(N_STORES)) + list(
        np.random.choice(
            np.arange(N_STORES),
            max_contributions - N_STORES,
            replace=True,
            p=store_id_contribution_pb,
        )
    )
    users_and_stores.extend(list(zip(umax, stores_affected_by_umax)))

    # For other users, assign up to max_contributions
    remaining_transactions = n_transactions - max_contributions
    other_user_ids = user_ids[user_ids != user_with_max_contributions]

    while remaining_transactions > 0:
        # Choose a random user that hasn't reached max_contributions yet
        chosen_user = np.random.choice(other_user_ids)
        if user_contributions[chosen_user] < max_contributions:
            # Add one transaction to this user
            users_and_stores.append((chosen_user, np.random.randint(0, N_STORES)))
            user_contributions[chosen_user] += 1
            remaining_transactions -= 1

    np.random.shuffle(users_and_stores)

    users = [i[0] for i in users_and_stores]
    stores = [i[1] for i in users_and_stores]

    other_id = np.arange(N_OTHER_GROUPS)
    user2other = {u: np.random.choice(other_id) for u in users}

    transactions_df["user_id"] = users
    transactions_df["store_id"] = stores
    transactions_df["other_id"] = [user2other[u] for u in users]
    return transactions_df


def generate_adj_datasets(new_schema: str, user_id: int):
    """Generate adjacent by removing 1 user (user_id) from the dataset tables.
    The new dataset is pushed under the new_schema.
    """
    engine = db_engine()
    Session = sessionmaker(bind=engine)

    original_schema = "public"
    metadata = MetaData(schema=original_schema)
    user_table = Table("users", metadata, autoload_with=engine)
    trans_table = Table("transactions", metadata, autoload_with=engine)

    with engine.begin() as conn:
        if not engine.dialect.has_schema(conn, new_schema):
            conn.execute(CreateSchema(new_schema))

    with Session() as session:
        filtered_users = session.execute(
            select(user_table).where(user_table.c.id != str(user_id))
        ).all()

        filtered_transactions = session.execute(
            select(trans_table).where(trans_table.c.user_id != str(user_id))
        ).all()

    (new_user_table, new_trans_table) = create_ds_from_metadata(new_schema)

    with engine.begin() as conn:
        # Insert into new_user_table
        for row in filtered_users:
            conn.execute(new_user_table.insert().values(**row._mapping))

        # Insert into new_trans_table
        for row in filtered_transactions:
            conn.execute(new_trans_table.insert().values(**row._mapping))


def main():
    generate_D_0_dataset()
    generate_adj_datasets(D_1, user_id=0)


if __name__ == "__main__":
    main()
