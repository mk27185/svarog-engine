"""
Terrain stamping: modifies the elevation grid so terrain smoothly
blends toward road centerlines.

Algorithm
---------
1. Each road polyline is subdivided into points every SUBDIVISION_STEP metres.
2. At every road point the current terrain elevation is sampled (bilinear).
3. The Z profile of each road is smoothed with a moving-average window to
   remove high-frequency SRTM noise (roads don't jerk up/down every 30 m).
4. For each terrain vertex the nearest road point is found; if it falls
   within (road_width/2 + BLEND_ZONE) the vertex elevation is blended
   toward the road elevation, with weight 1.0 inside the road footprint
   and linearly falling to 0 at the blend boundary.

GPU acceleration
----------------
When CuPy is available the pairwise distance matrix (N_vertices × M_road_points)
is computed on the GPU.  Falls back to NumPy automatically.
"""

import math
import numpy as np
from scipy.ndimage import uniform_filter1d

from .geo_utils import build_meta, to_local, sample_z

# Road half-widths by OSM highway tag (metres)
ROAD_HALF_WIDTHS: dict[str, float] = {
    "motorway": 7.0,       "motorway_link": 3.5,
    "trunk": 6.0,          "trunk_link": 3.0,
    "primary": 5.0,        "primary_link": 2.5,
    "secondary": 4.0,      "secondary_link": 2.0,
    "tertiary": 3.0,       "tertiary_link": 1.5,
    "unclassified": 2.5,   "residential": 2.5,
    "service": 1.5,        "living_street": 2.0,
    "pedestrian": 2.0,     "track": 1.5,
    "footway": 0.75,       "path": 0.75,
    "cycleway": 1.0,       "steps": 1.0,
    "bridleway": 1.25,
}
DEFAULT_HALF_WIDTH = 2.5   # metres

SUBDIVISION_STEP = 1.5     # metres between subdivided road points
SMOOTH_WINDOW    = 30      # samples for moving-average (~45 m at 1.5 m step)
BLEND_ZONE       = 6.0     # metres beyond road edge for terrain blending
GPU_BATCH        = 2_000   # road points per GPU batch — safe for 24 GB VRAM at 10× upsample


class TerrainStamper:
    """
    Stamps road geometry into the terrain z_grid.

    Parameters
    ----------
    use_gpu:          Try to use CuPy (RTX 3090 / any CUDA device).
    subdivision_step: Metres between interpolated road points.
    blend_zone:       Metres beyond road edge where terrain is blended.
    """

    def __init__(
        self,
        use_gpu: bool = True,
        subdivision_step: float = SUBDIVISION_STEP,
        blend_zone: float = BLEND_ZONE,
    ):
        self.subdivision_step = subdivision_step
        self.blend_zone = blend_zone
        self._xp, self._gpu = self._init_backend(use_gpu)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stamp(
        self,
        z_grid: np.ndarray,
        meta: dict,
        highways: list[dict],
    ) -> np.ndarray:
        """
        Returns stamped z_grid (h+1, w+1) with terrain blended toward road levels.
        """
        h, w = meta["h"], meta["w"]

        # Build flat vertex position arrays (metric)
        rows = np.arange(h + 1, dtype=np.float32)
        cols = np.arange(w + 1, dtype=np.float32)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        vx = cc * meta["cell_width_m"]
        vy = (h - rr) * meta["cell_height_m"]          # r=0→north→large y

        N = (h + 1) * (w + 1)
        vertex_xy = np.stack([vx.ravel(), vy.ravel()], axis=1)  # (N,2)
        vertex_z  = z_grid.ravel().astype(np.float32)            # (N,)

        # Build road point arrays
        road_xy_list, road_z_list, road_hw_list = [], [], []

        for road in highways:
            nodes = road["nodes"]
            if len(nodes) < 2:
                continue

            tags     = road.get("tags", {})
            hw_type  = tags.get("highway", "")
            half_w   = self._half_width(tags, hw_type)

            local_nodes = [to_local(lon, lat, meta) for lon, lat in nodes]
            pts         = self._subdivide(local_nodes)
            if len(pts) < 1:
                continue

            pts_arr = np.array(pts, dtype=np.float32)

            z_raw    = np.array([sample_z(x, y, z_grid, meta) for x, y in pts],
                                dtype=np.float32)
            z_smooth = self._smooth(z_raw)

            road_xy_list.append(pts_arr)
            road_z_list.append(z_smooth)
            road_hw_list.append(np.full(len(pts_arr), half_w, dtype=np.float32))

        if not road_xy_list:
            print("   TerrainStamper: žádné silnice k stamp-ování.")
            return z_grid

        road_xy = np.concatenate(road_xy_list, axis=0)
        road_z  = np.concatenate(road_z_list,  axis=0)
        road_hw = np.concatenate(road_hw_list, axis=0)

        print(
            f"   TerrainStamper: {len(road_xy):,} bodů silnic → "
            f"{N:,} vrcholů terénu  "
            f"[{'GPU' if self._gpu else 'CPU'}]"
        )

        blended, _ = self._blend(vertex_xy, vertex_z, road_xy, road_z, road_hw)
        return blended.reshape(h + 1, w + 1)

    # ------------------------------------------------------------------
    # Blending kernel (GPU or CPU)
    # ------------------------------------------------------------------

    def _blend(
        self,
        vertex_xy: np.ndarray,   # (N, 2)
        vertex_z:  np.ndarray,   # (N,)
        road_xy:   np.ndarray,   # (M, 2)
        road_z:    np.ndarray,   # (M,)
        road_hw:   np.ndarray,   # (M,)  half-widths
    ) -> np.ndarray:
        xp = self._xp
        blend_zone = np.float32(self.blend_zone)

        vxy = xp.asarray(vertex_xy)
        vz  = xp.asarray(vertex_z)
        rxy = xp.asarray(road_xy)
        rz  = xp.asarray(road_z)
        rhw = xp.asarray(road_hw)

        N = len(vz)
        M = len(rz)

        # Track per-vertex: maximum influence weight and corresponding target Z
        max_weight   = xp.zeros(N, dtype=xp.float32)
        best_z       = vz.copy()
        arange_N     = xp.arange(N)

        for i in range(0, M, GPU_BATCH):
            b_xy = rxy[i : i + GPU_BATCH]   # (B, 2)
            b_z  = rz [i : i + GPU_BATCH]   # (B,)
            b_hw = rhw[i : i + GPU_BATCH]   # (B,)

            # Squared distances  (N, B)
            diff = vxy[:, None, :] - b_xy[None, :, :]   # (N, B, 2)
            dist = xp.sqrt((diff * diff).sum(axis=2))    # (N, B)

            # Weight: 1.0 inside road, linear falloff to 0 at blend boundary
            edge_dist = dist - b_hw[None, :]             # (N, B): neg = inside road
            weight    = xp.clip(
                1.0 - edge_dist / blend_zone, 0.0, 1.0
            )                                            # (N, B)

            # Best road point in this batch for each vertex
            best_idx = xp.argmax(weight, axis=1)         # (N,)
            best_w   = weight[arange_N, best_idx]        # (N,)
            target_z = b_z[best_idx]                     # (N,)

            improve    = best_w > max_weight
            max_weight = xp.where(improve, best_w,   max_weight)
            best_z     = xp.where(improve, target_z, best_z)

        result = vz * (1.0 - max_weight) + best_z * max_weight

        # Transfer GPU → CPU
        try:
            return result.get(), max_weight.get()   # max_weight kept for callers that want it
        except AttributeError:
            return np.asarray(result), np.asarray(max_weight)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _half_width(tags: dict, hw_type: str) -> float:
        try:
            w = float(tags.get("width", 0) or 0)
            if w > 0:
                return w / 2.0
        except (ValueError, TypeError):
            pass
        return ROAD_HALF_WIDTHS.get(hw_type, DEFAULT_HALF_WIDTH)

    def _subdivide(self, local_nodes: list[tuple]) -> list[tuple]:
        """Subdivide polyline so consecutive points are ≤ subdivision_step apart."""
        step = self.subdivision_step
        pts: list[tuple] = []
        for i in range(len(local_nodes) - 1):
            x0, y0 = local_nodes[i]
            x1, y1 = local_nodes[i + 1]
            L = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            if L < 1e-6:
                continue
            n = max(1, math.ceil(L / step))
            for j in range(n):
                t = j / n
                pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
        if local_nodes:
            pts.append(local_nodes[-1])
        return pts

    @staticmethod
    def _smooth(zs: np.ndarray) -> np.ndarray:
        if len(zs) < 3:
            return zs
        window = min(SMOOTH_WINDOW, len(zs))
        return uniform_filter1d(zs, size=window, mode="nearest").astype(np.float32)

    @staticmethod
    def _init_backend(use_gpu: bool):
        if use_gpu:
            try:
                # When CuPy + CUDA libs are installed via pip (nvidia-cuda-nvrtc-cu12)
                # the CUDA root is not in the default search path; set it automatically.
                import os, importlib.util
                if not os.environ.get("CUDA_PATH"):
                    nvrtc_spec = importlib.util.find_spec("nvidia.cuda_nvrtc")
                    if nvrtc_spec and nvrtc_spec.submodule_search_locations:
                        pkg_dir = list(nvrtc_spec.submodule_search_locations)[0]
                        lib_dir = os.path.join(pkg_dir, "lib")
                        cuda_root = os.path.dirname(pkg_dir)
                        os.environ["CUDA_PATH"] = cuda_root
                        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
                        os.environ["LD_LIBRARY_PATH"] = (
                            lib_dir + (":" + existing_ld if existing_ld else "")
                        )

                import cupy as cp
                cp.zeros(1, dtype=cp.float32).copy()  # probe: requires kernel compilation
                print("TerrainStamper: GPU backend (CuPy / CUDA)")
                return cp, True
            except Exception as exc:
                print(f"TerrainStamper: CuPy nedostupné ({exc}), přepínám na CPU.")
        print("TerrainStamper: CPU backend (NumPy)")
        return np, False
