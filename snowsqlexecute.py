from sqlalchemy import create_engine,text
from dotenv import load_dotenv
import os

load_dotenv()

username=os.getenv("SNOWFLAKE_USER_NAME")
password=os.getenv("SNOWFLAKE_PASSWORD")
account=os.getenv("SNOWFLAKE_ACCOUNT")
warehouse=os.getenv("SNOWFLAKE_WAREHOUSE")

def execute_query(query):
    engine = create_engine(
        f'snowflake://{username}:{password}@{account}/DATA_ANALYTICS/ECOMMERCEMART'
        f'?warehouse={warehouse}'
    )
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            columns = list(result.keys())
            return columns, rows
    finally:
        engine.dispose()

def get_schema():
    query = """select 
    TABLE_CATALOG,
    TABLE_SCHEMA,
    TABLE_NAME,
    COLUMN_NAME,
    DATA_TYPE 
    from DATA_ANALYTICS.INFORMATION_SCHEMA.COLUMNS 
    where TABLE_SCHEMA = 'ECOMMERCEMART'
    """
    schema_fetch = execute_query(query)
    return schema_fetch

#print(get_schema())
#print(execute_query("SELECT * FROM DATASET_STATISTICS"))