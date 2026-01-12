import numpy as np
import h3
import math
import random
from datetime import datetime, timedelta

MIN_LAT, MAX_LAT = 12.85, 13.25
MIN_LON, MAX_LON = 79.95, 80.35

H3_RES = 8
BUCKET_MIN = 5
WINDOW = 12
HORIZONS = [1, 3, 6]
DAYS = 180

NUM_CENTERS = 100
EVENT_PROB = 0.04

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

geojson = {
    "type": "Polygon",
    "coordinates": [[
        [MIN_LON, MIN_LAT],
        [MIN_LON, MAX_LAT],
        [MAX_LON, MAX_LAT],
        [MAX_LON, MIN_LAT],
        [MIN_LON, MIN_LAT]
    ]]
}

cells = list(h3.geo_to_cells(geojson, H3_RES))
cell_to_idx = {c: i for i, c in enumerate(cells)}
idx_to_cell = {i: c for c, i in cell_to_idx.items()}
N = len(cells)

A = np.zeros((N, N), dtype=np.uint8)
for c in cells:
    i = cell_to_idx[c]
    for n in h3.grid_disk(c, 1):
        if n in cell_to_idx:
            j = cell_to_idx[n]
            A[i, j] = 1
            A[j, i] = 1
np.fill_diagonal(A, 1)

center_cells = random.sample(cells, NUM_CENTERS)
center_coords = [h3.cell_to_latlng(c) for c in center_cells]

static = np.zeros((N, 4))  # dist_to_center, commercial, residential, transit

for i, cell in idx_to_cell.items():
    lat, lon = h3.cell_to_latlng(cell)

    min_dist = min(
        haversine(lat, lon, c_lat, c_lon)
        for c_lat, c_lon in center_coords
    )

    static[i] = [
        min_dist,
        int(random.random() < 0.3),
        int(random.random() < 0.5),
        int(random.random() < 0.2)
    ]

T = int(DAYS * 24 * 60 / BUCKET_MIN)
times = [datetime(2025, 1, 1) + timedelta(minutes=i*BUCKET_MIN) for i in range(T)]

search = np.zeros((N, T))
booking = np.zeros_like(search)
completed = np.zeros_like(search)
pickup = np.zeros_like(search)
dropoff = np.zeros_like(search)

base = np.random.randint(3, 8, size=N)
event_flags = np.random.rand(T) < EVENT_PROB

for t in range(T):
    hour = times[t].hour
    peak = 1.8 if hour in [8,9,17,18,19] else 0.8

    for i in range(N):
        s = base[i] * peak + np.random.poisson(1)
        if event_flags[t]:
            s *= 3

        search[i, t] = s
        booking[i, t] = s * random.uniform(0.35, 0.6)
        completed[i, t] = booking[i, t] * random.uniform(0.7, 0.9)
        pickup[i, t] = completed[i, t]
        dropoff[i, t] = pickup[i, t] * random.uniform(0.9, 1.1)

def diff(x): return np.diff(x, axis=1, prepend=0)

def ma3(x):
    y = np.zeros_like(x)
    for t in range(2, x.shape[1]):
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

search_to_booking = booking / (search + 1)
booking_to_pickup = pickup / (booking + 1)

neighbor_search = A @ search
neighbor_booking = A @ booking
neighbor_dropoff = A @ dropoff
distance_weighted_neighbor = neighbor_search / (A.sum(axis=1, keepdims=True) + 1)

hour_sin = np.array([math.sin(2*math.pi*t.hour/24) for t in times])
hour_cos = np.array([math.cos(2*math.pi*t.hour/24) for t in times])
dow_sin = np.array([math.sin(2*math.pi*t.weekday()/7) for t in times])
dow_cos = np.array([math.cos(2*math.pi*t.weekday()/7) for t in times])
is_weekend = np.array([int(t.weekday() >= 5) for t in times])
is_holiday = np.zeros(T)

hist_avg = pickup.mean(axis=1, keepdims=True)
hist_p90 = np.percentile(pickup, 90, axis=1, keepdims=True)
hist_var = pickup.var(axis=1, keepdims=True)
current_vs_hist = pickup / (hist_avg + 1)

X_list, Y_list = [], []

for t in range(WINDOW, T - max(HORIZONS)):
    X = np.stack([
        search[:,t-WINDOW:t], booking[:,t-WINDOW:t], completed[:,t-WINDOW:t],
        dropoff[:,t-WINDOW:t], pickup[:,t-WINDOW:t],
        search_to_booking[:,t-WINDOW:t], booking_to_pickup[:,t-WINDOW:t],
        search_vel[:,t-WINDOW:t], booking_vel[:,t-WINDOW:t],
        pickup_vel[:,t-WINDOW:t], dropoff_vel[:,t-WINDOW:t],
        search_acc[:,t-WINDOW:t], booking_acc[:,t-WINDOW:t],
        search_ma[:,t-WINDOW:t], booking_ma[:,t-WINDOW:t], pickup_ma[:,t-WINDOW:t],
        neighbor_search[:,t-WINDOW:t], neighbor_booking[:,t-WINDOW:t],
        neighbor_dropoff[:,t-WINDOW:t], distance_weighted_neighbor[:,t-WINDOW:t],
        np.tile(hour_sin[t-WINDOW:t], (N,1)),
        np.tile(hour_cos[t-WINDOW:t], (N,1)),
        np.tile(dow_sin[t-WINDOW:t], (N,1)),
        np.tile(dow_cos[t-WINDOW:t], (N,1)),
        np.tile(is_weekend[t-WINDOW:t], (N,1)),
        np.tile(is_holiday[t-WINDOW:t], (N,1)),
        np.tile(event_flags[t-WINDOW:t], (N,1)),
        np.tile(static[:,0:1], (1,WINDOW)),
        np.tile(static[:,1:2], (1,WINDOW)),
        np.tile(static[:,2:3], (1,WINDOW)),
        np.tile(static[:,3:4], (1,WINDOW)),
        np.tile(hist_avg, (1,WINDOW)),
        np.tile(hist_p90, (1,WINDOW)),
        np.tile(hist_var, (1,WINDOW)),
        current_vs_hist[:,t-WINDOW:t]
    ], axis=2)

    Y = np.stack([pickup[:, t+h] for h in HORIZONS], axis=1)

    X_list.append(X)
    Y_list.append(Y)

X = np.array(X_list)
Y = np.array(Y_list)

np.savez_compressed(
    "stgnn_dataset.npz",
    X=X,
    A=A,
    Y=Y
)

print("STGNN DATASET READY")
print("X:", X.shape)
print("A:", A.shape)
print("Y:", Y.shape)
print("Centers:", NUM_CENTERS)
