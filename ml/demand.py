from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
from dateutil import parser
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h3
import uvicorn


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-6
WINDOW = 12
NUM_CITY_CENTERS = 100

MIN_LAT, MAX_LAT = 12.8500, 13.2500
MIN_LON, MAX_LON = 79.9500, 80.3500
H3_RESOLUTION = 8

geojson_polygon = {
    "type": "Polygon",
    "coordinates": [[
        [MIN_LON, MIN_LAT],
        [MIN_LON, MAX_LAT],
        [MAX_LON, MAX_LAT],
        [MAX_LON, MIN_LAT],
        [MIN_LON, MIN_LAT]
    ]]
}

cells = sorted(list(h3.geo_to_cells(geojson_polygon, H3_RESOLUTION)))
cell_to_idx = {c: i for i, c in enumerate(cells)}
idx_to_cell = {i: c for c, i in cell_to_idx.items()}
N = len(cells)

adj = np.zeros((N, N), dtype=np.float32)
for c in cells:
    i = cell_to_idx[c]
    for nbr in h3.grid_ring(c, 1):
        if nbr in cell_to_idx:
            j = cell_to_idx[nbr]
            adj[i, j] = 1
            adj[j, i] = 1
np.fill_diagonal(adj, 1)

def normalize_adj(A):
    D = np.diag(1.0 / np.sqrt(A.sum(axis=1) + EPS))
    return D @ A @ D

ADJ = normalize_adj(adj)
ADJ_TENSOR = torch.tensor(ADJ, dtype=torch.float32, device=DEVICE)

print(f"Graph initialized with {N} nodes")

random.seed(42)
np.random.seed(42)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2 +
        math.cos(math.radians(lat1)) *
        math.cos(math.radians(lat2)) *
        math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


city_center_cells = set(random.sample(cells, NUM_CITY_CENTERS))
city_centers_latlon = [h3.cell_to_latlng(c) for c in city_center_cells]


distance_to_city_center = np.zeros((N, 1), dtype=np.float32)

for i, cell in enumerate(cells):
    lat, lon = h3.cell_to_latlng(cell)
    min_dist = float("inf")
    for clat, clon in city_centers_latlon:
        d = haversine_km(lat, lon, clat, clon)
        if d < min_dist:
            min_dist = d
    distance_to_city_center[i, 0] = min_dist


is_commercial_area = (
    np.exp(-distance_to_city_center / 5.0) >
    np.random.rand(N, 1)
).astype(np.float32)

is_residential_area = (
    np.exp(-distance_to_city_center / 12.0) >
    np.random.rand(N, 1)
).astype(np.float32)

is_near_transit_hub = (
    np.random.rand(N, 1) < 0.08
).astype(np.float32)

static_features = np.concatenate(
    [
        distance_to_city_center,  # 0
        is_commercial_area,       # 1
        is_residential_area,      # 2
        is_near_transit_hub       # 3
    ],
    axis=1
).astype(np.float32)

hist_avg = np.random.uniform(2.0, 5.0, size=(N, 1)).astype(np.float32)
hist_p90 = hist_avg * np.random.uniform(1.3, 1.6, size=(N, 1)).astype(np.float32)
hist_var = np.random.uniform(0.5, 2.0, size=(N, 1)).astype(np.float32)

class GraphConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return output + self.bias

class SpatialModule(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gcn = GraphConvolution(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        out = self.gcn(x, adj)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out

class TemporalModule(nn.Module):
    def __init__(self, channels: int, kernel_sizes: list = [7, 9]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                channels,
                channels // len(kernel_sizes),
                kernel_size=k,
                padding=(k - 1) // 2
            )
            for k in kernel_sizes
        ])
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, nodes, time, features = x.shape
        x_reshaped = x.permute(0, 1, 3, 2).reshape(batch * nodes, features, time)
        
        outputs = [conv(x_reshaped) for conv in self.convs]
        out = torch.cat(outputs, dim=1)
        out = out.reshape(batch, nodes, features, time).permute(0, 1, 3, 2)
        
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out

class TemporalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.time_encoder = nn.Sequential(
            nn.Linear(6, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU()
        )
        self.implicit_history_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, time_features: torch.Tensor):
        time_emb = self.time_encoder(time_features)
        implicit_hist = self.implicit_history_predictor(time_emb)
        return time_emb, implicit_hist

class ImplicitHistoricalProjection(nn.Module):
    def __init__(self, time_embedding_dim: int, feature_dim: int):
        super().__init__()
        self.node_projector = nn.Sequential(
            nn.Linear(time_embedding_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, implicit_hist: torch.Tensor, num_nodes: int):
        hist_features = self.node_projector(implicit_hist)
        return hist_features.unsqueeze(1).expand(-1, num_nodes, -1)

class AdaptiveFusion(nn.Module):
    def __init__(self, feature_dim: int, time_embedding_dim: int = 64):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(time_embedding_dim, feature_dim),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, recent_features, temporal_context, time_embedding):
        gate = self.gate_network(time_embedding).unsqueeze(1)
        combined = torch.cat([
            recent_features * gate,
            temporal_context * (1 - gate)
        ], dim=-1)
        return self.fusion(combined)

class STGNN(nn.Module):
    def __init__(
        self, 
        num_nodes: int,
        in_features: int,
        hidden_channels: int = 64,
        out_horizons: int = 12,
        num_blocks: int = 6,
        use_temporal_conditioning: bool = True
    ):  
        super().__init__()
        
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.hidden_channels = hidden_channels
        self.out_horizons = out_horizons
        self.use_temporal_conditioning = use_temporal_conditioning
        
        self.input_proj = nn.Linear(in_features, hidden_channels)
        
        if use_temporal_conditioning:
            self.time_embedding = TemporalEmbedding(64)
            self.implicit_hist_proj = ImplicitHistoricalProjection(64, hidden_channels)
            self.adaptive_fusion = AdaptiveFusion(hidden_channels, 64)
        
        self.modules_list = nn.ModuleList()
        for i in range(num_blocks):
            if i % 2 == 0:
                self.modules_list.append(TemporalModule(hidden_channels))
            else:
                self.modules_list.append(SpatialModule(hidden_channels))
        
        self.skip_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.1) for _ in range(num_blocks)
        ])
        
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_channels, 1),
            nn.Softmax(dim=2)
        )
        
        self.output_proj_multi = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, out_horizons)
        )
        
        self.output_proj_eta = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, 1)
        )
        
    def forward(self, x, adj, eta_time_features=None):
        batch, nodes, time, features = x.shape
        
        x = self.input_proj(x)
        input_state = x
        current = x
        
        for i, module in enumerate(self.modules_list):
            if isinstance(module, SpatialModule):
                outputs_t = []
                for t in range(time):
                    outputs_t.append(module(current[:, :, t, :], adj))
                current = torch.stack(outputs_t, dim=2)
            else:
                current = module(current)
            
            current = current + self.skip_weights[i] * input_state
        
        attention_weights = self.temporal_attention(current)
        recent_features = torch.sum(current * attention_weights, dim=2)
        
        if eta_time_features is None:
            output = self.output_proj_multi(recent_features)
        else:
            time_emb, implicit_hist = self.time_embedding(eta_time_features)
            temporal_context = self.implicit_hist_proj(implicit_hist, nodes)
            fused_features = self.adaptive_fusion(recent_features, temporal_context, time_emb)
            output = self.output_proj_eta(fused_features)
        
        return output

def temporal_features(dt: datetime):
    return np.array([
        math.sin(2 * math.pi * dt.hour / 24),
        math.cos(2 * math.pi * dt.hour / 24),
        math.sin(2 * math.pi * dt.weekday() / 7),
        math.cos(2 * math.pi * dt.weekday() / 7),
        float(dt.weekday() >= 5),
        0.0
    ], dtype=np.float32)


checkpoint = torch.load(
    "best_model.pt",
    map_location="cpu",
    weights_only=False
)

cfg = checkpoint["config"]

model = STGNN(
    num_nodes=cfg["num_nodes"],
    in_features=cfg["in_features"],
    hidden_channels=cfg["hidden_channels"],
    out_horizons=cfg["out_horizons"],
    use_temporal_conditioning=cfg["use_temporal_conditioning"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

WINDOW = 12  

app = FastAPI(title="STGNN Demand Inference API")


class NodeHistory(BaseModel):
    search: List[float]
    booking: List[float]
    completed: List[float]
    pickup: List[float]
    dropoff: List[float]

class NextHourRequest(BaseModel):
    timestamps: List[str]                 
    history: Dict[str, NodeHistory]



class EtaRequest(NextHourRequest):
    eta_datetime: str


# class EtaWithAnchorsRequest(NextHourRequest):
#     eta_datetime: str
#     historical_anchors: Dict[str, Dict[str, Optional[float]]]
#     hist_avg: Dict[str, float]
#     hist_var: Dict[str, float]


def build_features(req: NextHourRequest) -> np.ndarray:
    times = [parser.isoparse(t) for t in req.timestamps]

    search = np.stack([req.history[h].search for h in cells])
    booking = np.stack([req.history[h].booking for h in cells])
    completed = np.stack([req.history[h].completed for h in cells])
    pickup = np.stack([req.history[h].pickup for h in cells])
    dropoff = np.stack([req.history[h].dropoff for h in cells])

    def diff(x):
        return np.diff(x, axis=1, prepend=x[:, :1])

    def ma3(x):
        y = np.zeros_like(x)
        for t in range(WINDOW):
            if t < 2:
                y[:, t] = x[:, t]
            else:
                y[:, t] = (x[:, t] + x[:, t-1] + x[:, t-2]) / 3
        return y

    search_vel = diff(search)
    booking_vel = diff(booking)
    pickup_vel = diff(pickup)
    dropoff_vel = diff(dropoff)

    search_acc = diff(search_vel)
    booking_acc = diff(booking_vel)

    search_ma = ma3(search)
    booking_ma = ma3(booking)
    pickup_ma = ma3(pickup)

    search_to_booking = booking / (search + EPS)
    booking_to_pickup = pickup / (booking + EPS)

    current_vs_hist = pickup / (hist_avg + EPS)

    neighbor_search = ADJ @ search
    neighbor_booking = ADJ @ booking
    neighbor_dropoff = ADJ @ dropoff
    distance_weighted = neighbor_search / (ADJ.sum(axis=1, keepdims=True) + 1)

    hour_sin = np.tile([math.sin(2*math.pi*t.hour/24) for t in times], (N, 1))
    hour_cos = np.tile([math.cos(2*math.pi*t.hour/24) for t in times], (N, 1))
    dow_sin = np.tile([math.sin(2*math.pi*t.weekday()/7) for t in times], (N, 1))
    dow_cos = np.tile([math.cos(2*math.pi*t.weekday()/7) for t in times], (N, 1))
    is_weekend = np.tile([float(t.weekday() >= 5) for t in times], (N, 1))
    is_holiday = np.zeros((N, WINDOW), dtype=np.float32)
    event_flags = np.zeros((N, WINDOW), dtype=np.float32)

    static_0 = np.tile(static_features[:, 0:1], (1, WINDOW))
    static_1 = np.tile(static_features[:, 1:2], (1, WINDOW))
    static_2 = np.tile(static_features[:, 2:3], (1, WINDOW))
    static_3 = np.tile(static_features[:, 3:4], (1, WINDOW))

    hist_avg_t = np.tile(hist_avg, (1, WINDOW))
    hist_p90_t = np.tile(hist_p90, (1, WINDOW))
    hist_var_t = np.tile(hist_var, (1, WINDOW))

    features = [
        search, booking, completed, dropoff, pickup,
        search_to_booking, booking_to_pickup,
        search_vel, booking_vel, pickup_vel, dropoff_vel,
        search_acc, booking_acc,
        search_ma, booking_ma, pickup_ma,
        neighbor_search, neighbor_booking, neighbor_dropoff,
        distance_weighted,
        hour_sin, hour_cos, dow_sin, dow_cos,
        is_weekend, is_holiday, event_flags,
        static_0, static_1, static_2, static_3,
        hist_avg_t, hist_p90_t, hist_var_t,
        current_vs_hist
    ]

    X = np.stack(features, axis=2).astype(np.float32)
    assert X.shape == (N, WINDOW, 35)
    return X


# def anchor_correction(base, anchors, hist_avg, hist_var):
#     values = []
#     weights = {
#         "yesterday_same_time": 0.35,
#         "last_week_same_day_time": 0.30,
#         "last_month_same_date_time": 0.20,
#         "last_year_same_date_time": 0.15
#     }

#     for k, w in weights.items():
#         if anchors.get(k) is not None:
#             values.append((anchors[k], w))

#     if not values:
#         return base

#     y_anchor = sum(v * w for v, w in values) / sum(w for _, w in values)

#     z = (y_anchor - hist_avg) / (math.sqrt(hist_var) + EPS)
#     alpha = 1 / (1 + math.exp(-(abs(z) - 0.75)))

#     return (1 - alpha) * base + alpha * y_anchor


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@app.post("/predict/next-hour")
def predict_next_hour(req: NextHourRequest):
    try:
        # print(req)
        if len(req.timestamps) != WINDOW:
               raise HTTPException(400, "timestamps must be length 12")

        missing = [h for h in cells if h not in req.history]
        if missing:
            raise HTTPException(400, f"Missing cells: {missing[:5]}")

        X_np = build_features(req)
        X = torch.tensor(X_np).unsqueeze(0).to(DEVICE)

        print("MODEL expects in_features =", model.in_features)
        print("ACTUAL input features =", X.shape[-1])

        with torch.no_grad():
            preds = model(X, ADJ_TENSOR)[0].cpu().numpy()

        return {
            "predictions": {
                h3: preds[i].tolist()
                for i, h3 in enumerate(cells)
            }
        }


    except HTTPException:
        raise
    except Exception as e:
        print("Inference error:", e)
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/predict/eta")
def predict_eta(req: EtaRequest):
    try:
        if len(req.timestamps) != WINDOW:
            raise HTTPException(400, "timestamps must be length 12")

        missing = [h for h in cells if h not in req.history]
        if missing:
            raise HTTPException(400, f"Missing cells: {missing[:5]}")
            
        X_np = build_features(req)
        X = torch.tensor(X_np).unsqueeze(0).to(DEVICE)

        eta = torch.tensor(
            temporal_features(datetime.fromisoformat(req.eta_datetime))
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            preds = model(X, ADJ_TENSOR, eta)[0].cpu().numpy()
        
        # Calculate scaling factor based on recent historical data
        # Average pickup demand from the last few hours
        recent_avg = np.mean([req.history[h].pickup[-3:] for h in cells], axis=1)
        
        # Scale predictions proportionally
        SCALING_FACTOR = 150  # Start with this, tune based on results
        # Or use adaptive scaling:
        # scaling_factor = np.mean(recent_avg) / (np.mean(preds[:, 0]) + 1e-6)
        
        predictions = {}
        for i, h3 in enumerate(cells):
            scaled_pred = preds[i, 0] * SCALING_FACTOR
            # Optional: blend with recent average for stability
            final_pred = 0.7 * scaled_pred + 0.3 * recent_avg[i]
            predictions[h3] = float(max(0, final_pred))

        return {"predictions": predictions}

    except HTTPException:
        raise
    except Exception as e:
        print("ETA inference error:", e)
        raise HTTPException(status_code=500, detail=str(e))



# @app.post("/predict/eta-with-history")
# def predict_eta_with_history(req: EtaWithAnchorsRequest):
#     try:
#         # Build features from historical data
#         X_np = build_features(req)
#         X = torch.tensor(X_np).unsqueeze(0).to(DEVICE)

#         # Create temporal features for the ETA datetime
#         eta = torch.tensor(
#             temporal_features(datetime.fromisoformat(req.eta_datetime))
#         ).unsqueeze(0).to(DEVICE)

#         # Get base predictions for all nodes
#         with torch.no_grad():
#             base_preds = model(X, ADJ_TENSOR, eta)[0].cpu().numpy()  # Shape: (N, 1)

#         # Apply anchor correction to all nodes
#         predictions = {}
#         for i, h3 in enumerate(cells):
#             base_pred = base_preds[i, 0]
            
#             # Apply anchor correction for this node
#             final_pred = anchor_correction(
#                 base_pred,
#                 req.historical_anchors.get(h3, {}),  # Get anchors for this specific node
#                 req.hist_avg.get(h3, hist_avg[i, 0]),  # Get hist_avg for this node
#                 req.hist_var.get(h3, hist_var[i, 0])   # Get hist_var for this node
#             )
            
#             predictions[h3] = round(final_pred, 4)

#         return {
#             "predictions": predictions
#         }

#     except HTTPException:
#         raise
#     except Exception as e:
#         print("ETA+history inference error:", e)
#         raise HTTPException(status_code=500, detail=str(e))

    









if __name__ == "__main__":
    uvicorn.run("demand:app", host="0.0.0.0", port=8000)
