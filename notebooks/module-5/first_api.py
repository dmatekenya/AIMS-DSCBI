from fastapi import FastAPI

app = FastAPI()

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
