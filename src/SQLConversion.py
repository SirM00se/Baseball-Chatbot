import pandas as pd
import sqlite3

df = pd.read_csv("../data/vector_metadata.csv")

conn = sqlite3.connect("../baseball_vectors.db")

df.to_sql("rules", conn, if_exists="replace", index=False)

conn.close()
print("CSV saved as SQLite database!")