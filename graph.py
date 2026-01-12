import h3
import numpy as np


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
cells = h3.geo_to_cells(geojson_polygon, H3_RESOLUTION)

cell_to_idx = {cell: i for i, cell in enumerate(cells)}
idx_to_cell = {i: cell for cell, i in cell_to_idx.items()}

N = len(cells)
print(N)


adj = np.zeros((N, N), dtype=np.uint8)

for cell in cells:
    i = cell_to_idx[cell]
    neighbors = h3.grid_ring(cell, 1)
    
    for nbr in neighbors:
        if nbr in cell_to_idx:
            j = cell_to_idx[nbr]
            adj[i, j] = 1
            adj[j, i] = 1

np.fill_diagonal(adj, 1)


