from fastapi import FastAPI, Query, Path
from typing import Optional
from datetime import datetime



# Create FastAPI application
app = FastAPI(
    title="Hello World API",
    description="A simple API to learn FastAPI basics",
    version="1.0.0"
)


# ==========================================================   
# BASIC ENDPOINT - NO PARAMETERS- WILL SHOW "HELLO WORLD!" 
# =========================================================
@app.get("/")
def read_root():
    """Welcome message - simplest possible endpoint"""
    return {"message": "Welcome to Nyagatare"}


# ===================================================
# ENDPOINT WITH PATH PARAMETER
# ===================================================
# This means the URL includes a variable part (here: {name})
# Example: calling /hello/Alice will pass "Alice" as the `name` argument
@app.get("/hello/{name}")
def say_hello(name: str):
    """Say hello to a specific person"""
    return {"message": f"Hello {name}!"}


# ======================================================
# ENDPOINT WITH QUERY PARAMETERS
# ======================================================
# Unlike path parameters, query parameters are passed after the "?" in the URL.
# Example: /greet?name=Alice&age=25&city=Kigali
# - "name" is required (Query(...))
# - "age" is optional (defaults to None if not provided)
# - "city" is optional with a default value of "Unknown"
@app.get("/greet")
def greet_person(
    name: str = Query(..., description="Person's name"),
    age: Optional[int] = Query(None, description="Person's age"),
    city: Optional[str] = Query("Kigali", description="Person's city")
):
    """Greet a person with optional details"""
    greeting = f"Hello {name}!"
    
    if age:
        greeting += f" You are {age} years old."
    
    greeting += f" You're from {city}."
    
    return {
        "greeting": greeting,
        "name": name,
        "age": age,
        "city": city,
        "timestamp": datetime.now().isoformat()
    }