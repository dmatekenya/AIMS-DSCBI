import os
from typing import Optional, List
import pandas as pd
import psycopg2
from fastapi import FastAPI, Query, Path, HTTPException
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

DATA_PATH = "/Users/dmatekenya/My Drive (dmatekenya@gmail.com)/TEACHING/AIMS-DSCBI/data/tmp-db-data/cells.csv"

# 1. Static endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to my first API!"}

# 2. Path parameter endpoint
@app.get("/greet/{name}/{city}")
def greet_person(name: str, city: str):
    return {"message": f"Hello {name} from {city}!"}

# 3. Data-serving endpoint (Rwanda sample data)
@app.get("/rwanda")
def get_rwanda_data():
    sample_data = {
        "capital": "Kigali",
        "population_millions": 13.8,
        "official_languages": ["Kinyarwanda", "English", "French"],
        "currency": "Rwandan franc (RWF)",
        "famous_for": [
            "Mountain gorillas",
            "Land of a Thousand Hills",
            "Clean and green capital city"
        ]
    }
    return {"rwanda_info": sample_data}

# Sample in-memory dataset
data = [
    {
        "name": "Denise",
        "age": 30,
        "city": "Kigali",
        "country": "Rwanda",
        "marital_status": "Guess",
        "children": 2
    },
    {
        "name": "Donald",
        "age": 20,
        "city": "Kigali",
        "country": "Rwanda",
        "marital_status": "Happily Married",
        "children": 0
    }
]


@app.get("/trainees")
def get_trainee_info(
    name: str = Query(..., description="Person's name (required)"),
    marital_status: Optional[bool] = Query(False, description="Include marital status"),
    age: Optional[bool] = Query(False, description="Include age"),
    children: Optional[bool] = Query(False, description="Include number of children")
):
    """
    Return trainee info:
    - Always return name, city, and country
    - If marital_status=true, include marital_status
    - If age=true, include age
    """
    # Find the person
    person = next((p for p in data if p["name"].lower() == name.lower()), None)
    print(person)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    # Base response
    response = {
        "name": person["name"],
        "city": person["city"],
        "country": person["country"]
    }

    # Add optional fields
    if marital_status:
        response["marital_status"] = person["marital_status"]
    if age:
        response["age"] = person["age"]
    
    if children:
        response["number_of_children"] = person["children"]

    return {"Trainee-Info": response}


# ==========================================================
# EXERCISE-1 - EDIT trainees endpoint
# =========================================================
# add children parameter (bool) to include number of children
# if children=true, include number_of_children in response
# access the app and see the results for yourself!
# http://localhost:8000/trainees?name=Denise&marital_status=true&age=true&children=true
# http://localhost:8000/trainees?name=Donald&marital_status=true&age=true&children=true


# ==========================================================
# RETURNING DATA -ALL CELLS FROM CSV
# ==========================================================

@app.get("/cells")
def list_cells():
    df = pd.read_csv(DATA_PATH)

    dict_from_pandas = df.to_dict(orient="records")

    return dict_from_pandas


# ==========================================================
# RETURN CELLS BY PROVINCE - PATH PARAMETER
# =========================================================
@app.get("/cells/province/{province_name}")
def get_cells_by_province_path(province_name: str):
    """
    Filter cells by province using a PATH parameter
    Example:
      /cells/province/Kigali
    """
    df = pd.read_csv(DATA_PATH)

    filtered = df[df["province_name"].str.lower() == province_name.lower()]

    return filtered.to_dict(orient="records")

# ==========================================================
# RETURN CELLS BY PROVINCE - QUERY PARAMETER
# =========================================================
@app.get("/cells-by-province")
def get_cells_by_province_query(
    province_name: Optional[str] = Query(None, description="Optional province filter")
):
    """
    Filter cells by province using a QUERY parameter
    Example:
      /cells-by-province?province_name=Kigali
      /cells-by-province   (returns all provinces if not specified)
    """
    df = pd.read_csv(DATA_PATH)
    if province_name:
        df = df[df["province_name"].str.lower() == province_name.lower()]
    
    return df.to_dict(orient="records")

# ACCESS THE APP AND SEE THE RESULTS FOR YOURSELF!
# http://localhost:8000/cells
# http://localhost:8000/cells/province/Kigali
# http://localhost:8000/cells-by-province?province_name=Kigali
  


# ==========================================================
# EXERCISE-2 - RETURN POPULATION BY PROVINCE
# =========================================================
# 1. Use Path parameter to get population by province
# 2. Use Query parameter to get population by province
# Hint: use similar logic as cells by province above


# ==========================================================
# RETURNING DATA WITH A DATA MODEL (PYDANTIC)
# =========================================================
from pydantic import BaseModel
from typing import Optional, List


# =============================================================
# DEFINE PYDANTIC MODEL FOR CELL WHICH MATCHES CSV STRUCTURE
# =============================================================
class Cell(BaseModel):
    cell_id: str
    province_name: Optional[str] = None
    district_name: Optional[str] = None
    sector_name: Optional[str] = None
    cell_name: Optional[str] = None


# =============================================================
# DEFINE ENDPOINT TO RETURN CELLS USING THE DATA MODEL
# =============================================================
@app.get("/cells-with-data-model", response_model=List[Cell])
def list_cells():
    df = pd.read_csv(DATA_PATH)
    return df.to_dict(orient="records")




# ==========================================================
# RETURNING DATA FROM POSTGRESQL DATABASE
# =========================================================

# =============================================================
# Load environment variables from .env file
# =============================================================
load_dotenv()

# Get database connection details from environment variables
PGHOST = os.getenv("PGHOST")
PGPORT = os.getenv("PGPORT")
PGDATABASE = os.getenv("PGDATABASE")
PGUSER = os.getenv("PGUSER")
PGPASSWORD = os.getenv("PGPASSWORD")


# =============================================================
# HELPER FUNCTION TO CONNECT AND QUERY
# =============================================================
def run_query(sql: str, params: tuple = ()):
    conn = psycopg2.connect(
        host=PGHOST,
        port=PGPORT,
        dbname=PGDATABASE,
        user=PGUSER,
        password=PGPASSWORD,
    )
    try:
        df = pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()
    return df.to_dict(orient="records")

# =============================================================
# ENDPOINTS (NO DATA MODEL)
# =============================================================
@app.get("/cells-db")
def list_cells():
    """Return all rows from cells table"""
    sql = "SELECT * FROM cells"
    return run_query(sql)



# ============================================================
# EXERCISE-3 - RETURN POPULATION BY PROVINCE FROM DATABASE
# ============================================================
# 1. use Path parameter to get population by province from DB
# 2. name the endpoint /population/province/{province_name}

