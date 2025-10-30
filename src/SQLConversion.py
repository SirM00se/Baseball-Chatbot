import pandas as pd
from sqlalchemy import create_engine

# Read the CSV file into a DataFrame
df = pd.read_csv("../data/vector_metadata.csv")

# Create a SQLAlchemy engine for SQLite
engine = create_engine("sqlite:///../databases/baseball_vectors.db")

# Write the DataFrame to the database
df.to_sql("rules", con=engine, if_exists="replace", index=False)

# Dispose of the engine connection
engine.dispose()

print("CSV saved as SQLite database via SQLAlchemy!")