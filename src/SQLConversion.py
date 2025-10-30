import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError, ProgrammingError, IntegrityError, ArgumentError

# -----------------------------
# Load CSV Data
# -----------------------------
def load_csv_data(csv_path: str) -> pd.DataFrame:
    """Load data from a CSV file into a pandas DataFrame."""
    df = pd.read_csv(csv_path)
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df


# -----------------------------
# Create SQLite Engine
# -----------------------------
def create_sqlite_engine(db_path: str):
    """Create and return a SQLAlchemy SQLite engine."""
    try:
        engine = create_engine(f"sqlite:///{db_path}")
        # Test the connection immediately
        conn = engine.connect()
        conn.close()  # close test connection
        print(f"Connected to SQLite database at {db_path}")
        return engine
    except ArgumentError as e:
        print(f"Invalid database path or connection string: {e}")
        raise  # re-raise if connection is critical
    except OperationalError as e:
        print(f"Operational error (permissions, missing folder, etc.): {e}")
        raise
    except Exception as e:
        print(f"Unexpected error while creating SQLite engine: {e}")
        raise


# -----------------------------
# Write DataFrame to SQL
# -----------------------------
def write_to_database(df: pd.DataFrame, engine, table_name: str):
    """Write a DataFrame to an SQL table, replacing it if it exists."""
    try:
        df.to_sql(table_name, con=engine, if_exists="replace", index=False)
        print(f"Wrote DataFrame to table '{table_name}' ({len(df)} rows).")
    except IntegrityError as e:
        print(f"Integrity error: {e}")
        raise  # Fail-fast if data violates constraints
    except OperationalError as e:
        print(f"Operational error (connection/file issue): {e}")
        raise
    except ProgrammingError as e:
        print(f"SQL/Programming error (invalid table/column names): {e}")
        raise
    except SQLAlchemyError as e:
        print(f"General SQLAlchemy error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error while writing DataFrame to SQL: {e}")
        raise


# -----------------------------
# Close Database Connection
# -----------------------------
def close_connection(engine):
    """Dispose of the SQLAlchemy engine connection."""
    try:
        engine.dispose()
        print("Engine disposed successfully")
    except SQLAlchemyError as e:
        print(f"SQLAlchemy error while disposing engine: {e}")
    except Exception as e:
        print(f"Unexpected error while disposing engine: {e}")


# -----------------------------
# Main Execution
# -----------------------------
def main():
    csv_path = "../data/vector_metadata.csv"
    db_path = "../databases/baseball_vectors.db"
    table_name = "rules"

    # Step 1: Load CSV
    df = load_csv_data(csv_path)

    # Step 2: Create SQLite engine
    engine = create_sqlite_engine(db_path)

    # Step 3: Write DataFrame to SQL
    write_to_database(df, engine, table_name)

    # Step 4: Dispose connection
    close_connection(engine)

    print("CSV successfully saved as SQLite database via SQLAlchemy!")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    main()
