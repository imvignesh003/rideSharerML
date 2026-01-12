from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import uvicorn


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-6


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


class NextHourRequest(BaseModel):
    h3_indexes: List[str]
    pickup_history: Dict[str, List[float]]
    adjacency: List[List[float]]


class EtaRequest(BaseModel):
    h3_indexes: List[str]
    pickup_history: Dict[str, List[float]]
    adjacency: List[List[float]]
    h3_index: str
    eta_datetime: str


class EtaWithAnchorsRequest(EtaRequest):
    historical_anchors: Dict[str, Optional[float]]
    hist_avg: float
    hist_var: float


def anchor_correction(base, anchors, hist_avg, hist_var):
    values = []
    weights = {
        "yesterday_same_time": 0.35,
        "last_week_same_day_time": 0.30,
        "last_month_same_date_time": 0.20,
        "last_year_same_date_time": 0.15
    }

    for k, w in weights.items():
        if anchors.get(k) is not None:
            values.append((anchors[k], w))

    if not values:
        return base

    y_anchor = sum(v * w for v, w in values) / sum(w for _, w in values)

    z = (y_anchor - hist_avg) / (math.sqrt(hist_var) + EPS)
    alpha = 1 / (1 + math.exp(-(abs(z) - 0.75)))

    return (1 - alpha) * base + alpha * y_anchor


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@app.post("/predict/next-hour")
def predict_next_hour(req: NextHourRequest):
    try:
        h3_to_idx = {h: i for i, h in enumerate(req.h3_indexes)}
        pickup = np.stack([req.pickup_history[h] for h in req.h3_indexes])
        x = pickup[:, -WINDOW:][..., None]

        X = torch.tensor(x).unsqueeze(0).to(DEVICE)
        A = torch.tensor(req.adjacency, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            preds = model(X, A)[0].cpu().numpy()

        return {
            "predictions": {
                h3: preds[h3_to_idx[h3]].tolist()
                for h3 in req.h3_indexes
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/eta")
def predict_eta(req: EtaRequest):
    try:
        idx = req.h3_indexes.index(req.h3_index)
        pickup = np.stack([req.pickup_history[h] for h in req.h3_indexes])
        x = pickup[:, -WINDOW:][..., None]

        eta = torch.tensor(
            temporal_features(datetime.fromisoformat(req.eta_datetime))
        ).unsqueeze(0).to(DEVICE)

        X = torch.tensor(x).unsqueeze(0).to(DEVICE)
        A = torch.tensor(req.adjacency, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            pred = model(X, A, eta)[0, idx, 0].item()

        return {"h3_index": req.h3_index, "predicted_demand": round(pred, 4)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/eta-with-history")
def predict_eta_with_history(req: EtaWithAnchorsRequest):
    try:
        idx = req.h3_indexes.index(req.h3_index)
        pickup = np.stack([req.pickup_history[h] for h in req.h3_indexes])
        x = pickup[:, -WINDOW:][..., None]

        eta = torch.tensor(
            temporal_features(datetime.fromisoformat(req.eta_datetime))
        ).unsqueeze(0).to(DEVICE)

        X = torch.tensor(x).unsqueeze(0).to(DEVICE)
        A = torch.tensor(req.adjacency, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            base_pred = model(X, A, eta)[0, idx, 0].item()

        final_pred = anchor_correction(
            base_pred,
            req.historical_anchors,
            req.hist_avg,
            req.hist_var
        )

        return {
            "h3_index": req.h3_index,
            "predicted_demand": round(final_pred, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("demand:app", host="0.0.0.0", port=8000)
