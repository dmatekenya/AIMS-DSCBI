from fastapi import FastAPI, Query, Path
from typing import Optional
from datetime import datetime

# Create FastAPI application
app = FastAPI(
    title="Hello World API",
    description="A simple API to learn FastAPI basics",
    version="1.0.0"
)

# Basic endpoint - no parameters
@app.get("/")
def read_root():
    """Welcome message - simplest possible endpoint"""
    return {"message": "Hello World!"}

# Endpoint with path parameter
@app.get("/hello/{name}")
def say_hello(name: str):
    """Say hello to a specific person"""
    return {"message": f"Hello {name}!"}

# Endpoint with query parameters
@app.get("/greet")
def greet_person(
    name: str = Query(..., description="Person's name"),
    age: Optional[int] = Query(None, description="Person's age"),
    city: Optional[str] = Query("Unknown", description="Person's city")
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