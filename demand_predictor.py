# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import joblib
# import numpy as np
# from typing import List
# import uvicorn

# app = FastAPI(title="Demand Prediction API")

# # Load model at startup
# model = joblib.load('demand_predictor.pkl')
# feature_cols = joblib.load('feature_columns.pkl')

# class DemandRequest(BaseModel):
#     location_lat: float
#     location_lon: float
#     hour_of_day: int = 18
#     day_of_week: int = 2
#     is_weekend: int = 0
#     is_peak_hour: int = 1
#     pickup_requests_last_1hr: int = 100
#     pickup_requests_last_24hr: int = 1500
#     successful_pickups_last_7days: int = 3000
#     is_commercial_area: int = 1
#     is_residential_area: int = 0
#     is_near_transit_hub: int = 0
#     distance_to_city_center_km: float = 2.0

# class DemandResponse(BaseModel):
#     demand_score: float

# class BatchDemandRequest(BaseModel):
#     locations: List[DemandRequest]

# @app.post("/predict-demand", response_model=DemandResponse)
# def predict_demand(request: DemandRequest):
#     """
#     Predict demand for a single location
    
#     Example:
#     POST /predict-demand
#     {
#         "location_lat": 13.0827,
#         "location_lon": 80.2707,
#         "hour_of_day": 9,
#         "pickup_requests_last_1hr": 25
#     }
#     """
#     try:
#         # Prepare features
#         features = [
#             request.location_lat,
#             request.location_lon,
#             request.hour_of_day,
#             request.day_of_week,
#             request.is_weekend,
#             request.is_peak_hour,
#             request.pickup_requests_last_1hr,
#             request.pickup_requests_last_24hr,
#             request.successful_pickups_last_7days,
#             request.is_commercial_area,
#             request.is_residential_area,
#             request.is_near_transit_hub,
#             request.distance_to_city_center_km
#         ]
        
#         # Predict
#         demand_score = float(model.predict([features])[0])
#         demand_score = np.clip(demand_score, 0, 1)
        
#         # Calculate wait time
#         wait_time = 60 * (1 - demand_score)
        
#         return DemandResponse(
#             demand_score=round(demand_score, 4),
#             pickup_probability=round(demand_score, 4),
#             expected_wait_time_minutes=round(wait_time, 1)
#         )
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/predict-demand-batch")
# def predict_demand_batch(request: BatchDemandRequest):
#     """
#     Predict demand for multiple locations (batch)
    
#     Faster than multiple single requests
#     """
#     try:
#         results = []
        
#         for loc in request.locations:
#             features = [
#                 loc.location_lat, loc.location_lon, loc.hour_of_day,
#                 loc.day_of_week, loc.is_weekend, loc.is_peak_hour,
#                 loc.pickup_requests_last_1hr, loc.pickup_requests_last_24hr,
#                 loc.successful_pickups_last_7days, loc.avg_wait_time_at_location,
#                 loc.is_commercial_area, loc.is_residential_area,
#                 loc.is_near_transit_hub, loc.distance_to_city_center_km
#             ]
            
#             demand = float(model.predict([features])[0])
#             demand = np.clip(demand, 0, 1)
            
#             results.append({
#                 'lat': loc.location_lat,
#                 'lon': loc.location_lon,
#                 'demand_score': round(demand, 4),
#                 'pickup_probability': round(demand, 4),
#                 'expected_wait_time_minutes': round(60 * (1 - demand), 1)
#             })
        
#         return {'predictions': results}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/health")
# def health_check():
#     """Health check endpoint"""
#     return {"status": "healthy", "model_loaded": True}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)