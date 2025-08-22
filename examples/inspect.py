import duckdb

# Connect to the database
db_path = "/home/bester/data/test_ascii_1h60.0s.duckdb/data.duckdb"
conn = duckdb.connect(db_path)

print(f"=== Inspection Report for {db_path} ===\n")

# 1. List all tables and views
print("1. ALL TABLES AND VIEWS:")
tables_list = conn.execute("""
    SELECT table_name, table_type
    FROM information_schema.tables
    WHERE table_schema = 'main'
    ORDER BY table_type, table_name;
""").fetchall()

for table in tables_list:
    print(f"   * {table[0]} ({table[1]})")

# 2. Describe the structure of each table
print("\n2. TABLE STRUCTURES:")
for table in tables_list:
    table_name = table[0]
    print(f"\n   -- {table_name} --")
    columns = conn.execute(f"PRAGMA table_info('{table_name}');").fetchall()
    for col in columns:
        print(f"      {col[1]:<20} {col[2]}")

# 3. Show the definition of the main view (if it exists)
print("\n3. VIEW DEFINITION:")
try:
    view_def = conn.execute(
        "SELECT sql FROM duckdb_views() WHERE view_name = 'main_view';"
    ).fetchone()
    if view_def:
        print("   main_view SQL:")
        print("   ", view_def[0])
    else:
        print("   No view named 'main_view' found.")
except Exception as e:
    print(f"   Error retrieving view: {e}")

# 4. Show a sample from the MAIN data (assuming the view or table is called 'main' or 'main_view')
print("\n4. SAMPLE DATA FROM MAIN:")
result = conn.execute(
    "SELECT TIME, ANTENNA1, ANTENNA2, UVW, DATA_REAL, DATA_IMAG FROM main_view LIMIT 15;"
).fetchall()
for r in result:
    print("time = ", r[0])
    print("\n")
    print("ant1 ant2", result[1], result[2])
    print("\n")
    print("uvw = ", result[3])
    print("\n")
    print("data.real = ", result[4])
    print("\n")
    print("data.imag = ", result[5])
    print("\n")
conn.close()
print("\n=== Inspection Complete ===")
