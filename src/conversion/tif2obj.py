import numpy as np
import rasterio
from scipy.interpolate import RegularGridInterpolator
import os
import sys

def tif_to_obj(tif_path, obj_path):
    """
    Převádí GeoTIFF (DEM) na 3D OBJ mesh.
    """
    if not os.path.exists(tif_path):
        print(f"Chyba: Soubor {tif_path} neexistuje.")
        return

    print(f"Načítám: {tif_path}")
    with rasterio.open(tif_path) as src:
        elevation = src.read(1)
        transform = src.transform
        rows, cols = elevation.shape
        
        # Počet vrcholů (corners) v mřížce
        grid_rows = rows + 1
        grid_cols = cols + 1
        
        # 1. Vytvoření indexů vrcholů
        r_idx, c_idx = np.meshgrid(np.arange(grid_rows), np.arange(grid_cols), indexing='ij')
        v_rows = r_idx.flatten()
        v_cols = c_idx.flatten()
        
        # 2. Transformace pixelových souřadnic na globální (X, Y)
        # Transformace transformuje (col, row) -> (x, y)
        v_coords = np.array([transform * (c, r) for c, r in zip(v_cols, v_rows)])
        X = v_coords[:, 0]
        Y = v_coords[:, 1]
        
        # 3. Interpolace výšky (Z) pro vrcholy z pixelových středů
        # Středy pixelů jsou v (r + 0.5, c + 0.5)
        # Interpolátor definujeme na mřížce středů
        interp = RegularGridInterpolator(
            (np.arange(rows) + 0.5, np.arange(cols) + 0.5), 
            elevation, 
            method='linear', 
            bounds_error=False, 
            fill_value=0
        )
        
        # Interpolujeme výšku pro naše vrcholy (v_rows, v_cols)
        # Musíme ale interpolovat relativně k center-gridu (což je v našem případě přímo v indexech)
        # Protože interpolátor očekává body v rámci svých definovaných os
        # Stačí posunout body o -0.5, abychom se dostali ze vrcholů na středy
        Z = interp(np.stack([v_rows - 0.5, v_cols - 0.5], axis=-1))
        
        # 4. Tvorba trojúhelníků (faces)
        # Pro každý pixel (r, c) vytvoříme 2 trojúhelníky
        # Vertex indexy v plochém poli: idx = r * grid_cols + c
        r_range = np.arange(rows)
        c_range = np.arange(cols)
        rr, cc = np.meshgrid(r_range, c_range, indexing='ij')
        
        v0 = (rr * grid_cols + cc).flatten()
        v1 = (rr * grid_cols + (cc + 1)).flatten()
        v2 = ((rr + 1) * grid_cols + cc).flatten()
        v3 = ((rr + 1) * grid_cols + (cc + 1)).flatten()
        
        # Spojení obou trojúhelníků pro každý čtverec
        faces_1 = np.stack((v0, v1, v2), axis=-1)
        faces_2 = np.array([v1, v3, v2]).T
        all_faces = np.vstack((faces_1, faces_2))
        
        # 5. Export do OBJ
        print(f"Exportuji: {obj_path}")
        with open(obj_path, 'w') as f:
            f.write(f"# OBJ mesh from {os.path.basename(tif_path)}\n")
            # Vertices
            for i in range(len(X)):
                f.write(f"v {X[i]} {Y[i]} {Z[i]}\n")
            # Faces (OBJ používá 1-based indexing)
            for face in all_faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        print(f"Hotovo! Vertices: {len(X)}, Faces: {len(all_faces)}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Použití: python tif2obj.py <input_tif> <output_obj>")
    else:
        tif_to_obj(sys.argv[1], sys.argv[2])
