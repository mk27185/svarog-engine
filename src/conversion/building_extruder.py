import os
import math

from .geo_utils import to_local, sample_z

DEFAULT_BUILDING_HEIGHT = 10.0
LEVEL_HEIGHT = 3.0


class BuildingExtruder:
    """
    Extrudes OSM building footprints into 3D OBJ geometry.

    Ground elevation is sampled from the supplied z_grid (which may be the
    stamped grid so buildings at road intersections sit at road level).
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        buildings: list[dict],
        z_grid,
        meta: dict,
        *,
        obj_name: str | None = None,
        output_dir: str | None = None,
    ) -> str:
        """Plug-in compatible interface (BuildingPlugin protocol)."""
        return self.extrude_buildings(
            buildings, z_grid, meta,
            obj_name=obj_name, output_dir=output_dir,
        )

    def extrude_buildings(
        self,
        buildings: list[dict],
        z_grid,           # np.ndarray (h+1, w+1) — stamped or raw
        meta: dict,       # from TerrainConverter.build_grid() / geo_utils.build_meta()
        obj_name: str | None = None,
        output_dir: str | None = None,
        # Legacy compatibility: still accept tif_path but ignore it when
        # z_grid + meta are provided.
        tif_path: str | None = None,
    ) -> str:
        """
        Extrude building footprints and write an OBJ file.

        When z_grid + meta are provided they are used for height sampling.
        When only tif_path is provided (legacy), rasterio is used instead.
        """
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Legacy path: load grid from TIF if caller didn't supply z_grid
        if z_grid is None:
            if tif_path is None:
                raise ValueError("Provide either z_grid+meta or tif_path.")
            import rasterio
            import numpy as np
            from .geo_utils import build_meta
            from .terrain_converter import TerrainConverter
            z_grid, meta = TerrainConverter().build_grid(tif_path)

        all_vertices: list[tuple[float, float, float]] = []
        all_faces:    list[tuple[int, int, int]]       = []
        skipped = 0

        for building in buildings:
            nodes = building["nodes"]
            tags  = building.get("tags", {})

            if len(nodes) < 3:
                skipped += 1
                continue

            height = self._building_height(tags)

            # Sample terrain at every footprint vertex; use minimum so the
            # building never floats above the highest ground corner.
            corner_zs = [
                sample_z(*to_local(n["lon"], n["lat"], meta), z_grid, meta)
                for n in nodes
            ]
            base_z = min(corner_zs)
            top_z   = base_z + height
            n_nodes = len(nodes)

            base_start = len(all_vertices)
            for n in nodes:
                x, y = to_local(n["lon"], n["lat"], meta)
                all_vertices.append((x, y, base_z))
            top_start = len(all_vertices)
            for n in nodes:
                x, y = to_local(n["lon"], n["lat"], meta)
                all_vertices.append((x, y, top_z))

            # Walls
            for i in range(n_nodes):
                j  = (i + 1) % n_nodes
                b0 = base_start + i + 1
                b1 = base_start + j + 1
                t0 = top_start  + i + 1
                t1 = top_start  + j + 1
                all_faces.append((b0, b1, t0))
                all_faces.append((b1, t1, t0))

            # Flat roof — fan triangulation
            for i in range(1, n_nodes - 1):
                all_faces.append((
                    top_start + 1,
                    top_start + i + 1,
                    top_start + i + 2,
                ))

        suffix = f"{obj_name}_buildings" if obj_name else "buildings"
        out = os.path.join(output_dir, f"{suffix}.obj")

        with open(out, "w") as f:
            f.write("# Building OBJ – local metric coords\n")
            f.write(f"# Origin: lon={meta['lon_origin']:.6f}, lat={meta['lat_origin']:.6f}\n")
            f.write(f"# Buildings: {len(buildings) - skipped} extruded, {skipped} skipped\n")
            for v in all_vertices:
                f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
            for fc in all_faces:
                f.write(f"f {fc[0]} {fc[1]} {fc[2]}\n")

        print(
            f"   Budovy zapsány: {out}  "
            f"({len(all_vertices)} vrcholů, {len(all_faces)} faces)"
        )
        return out

    # ------------------------------------------------------------------

    @staticmethod
    def _building_height(tags: dict) -> float:
        if "height" in tags:
            try:
                return float(tags["height"].replace(",", ".").split()[0])
            except (ValueError, IndexError):
                pass
        if "building:levels" in tags:
            try:
                return int(tags["building:levels"]) * LEVEL_HEIGHT
            except ValueError:
                pass
        return DEFAULT_BUILDING_HEIGHT
