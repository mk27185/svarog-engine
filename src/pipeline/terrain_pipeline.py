"""
Full terrain pipeline
=====================
  1. DEM download (OpenTopography)
  2. Terrain grid  – bilinear upsample, smooth SRTM grid
  3. Terrain OBJ   – regular quad mesh with UV coords
  4. OSM highways  – download or load from cache
  5. SDF road texture – RGBA PNG (R=distance, G=rank, B=width, A=mask)
  6. Buildings OBJ – extruded footprints (OSM S3DB building:part support)
  7. GLB export    – combined terrain + buildings

Outputs (all in output_dir):
  <base>_terrain.obj       ← terrain mesh with UV coords
  <base>_roads_sdf.png     ← SDF road texture (apply in Three.js as terrain material)
  <base>_buildings.obj     ← extruded OSM buildings
  <base>.glb               ← combined GLB (terrain + buildings), Y-up
  <base>_osm_highways.json ← cached OSM highway data (re-used on subsequent runs)
  <base>_osm_buildings.json← cached OSM building data (re-used on subsequent runs)

OSM cache
---------
After the first successful download the highway and building data are saved as
JSON files next to the other tile outputs.  Subsequent pipeline runs for the
same tile load from these files instead of querying Overpass API, which saves
time and avoids rate-limiting.  Pass ``osm_force_download=True`` (or the CLI
flag ``--force-osm``) to bypass the cache and re-fetch from Overpass.

Plug-in contract
----------------
`building_extruder` must satisfy BuildingPlugin (has `generate(buildings, z_grid, meta, *, obj_name, output_dir)`).
"""
from __future__ import annotations
import json
import os
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.conversion.plugin import BuildingPlugin
    from src.conversion.gltf_exporter import GltfExporter

UPSAMPLE_FACTOR = 10   # 30 m SRTM → ≈ 3 m grid  (try 20 for ≈ 1.5 m)


class TerrainPipeline:

    def __init__(
        self,
        client,
        converter,
        osm_client                                 = None,
        building_extruder: "BuildingPlugin | None" = None,
        sdf_generator                              = None,
        gltf_exporter:     "GltfExporter | None"   = None,
        upsample_factor:   int                     = UPSAMPLE_FACTOR,
        osm_force_download: bool                   = False,
        # kept for backward compat with tests that still pass road_mesh
        road_mesh                                  = None,
    ):
        self.client             = client
        self.converter          = converter
        self.osm_client         = osm_client
        self.building_extruder  = building_extruder
        self.sdf_generator      = sdf_generator
        self.gltf_exporter      = gltf_exporter
        self.upsample_factor    = upsample_factor
        self.osm_force_download = osm_force_download

    def run_pipeline(self, bbox: tuple, output_base_name: str) -> dict:
        print(f"\n{'='*60}")
        print(f"  Pipeline: '{output_base_name}'  bbox={bbox}")
        print(f"  Upsample: {self.upsample_factor}×")
        print(f"{'='*60}")

        result = {
            "terrain":     None,
            "sdf_texture": None,
            "buildings":   None,
            "glb":         None,
        }

        # ── 1. DEM ────────────────────────────────────────────────────────
        print("\n[1/5] Stahuji DEM...")
        tif_path = self.client.get_dem(bbox)
        if not tif_path or not os.path.exists(tif_path):
            print("  ✗ Nepodařilo se stáhnout DEM.")
            return result
        print(f"  ✓ {tif_path}")

        # ── 2. Terrain grid ───────────────────────────────────────────────
        print(f"\n[2/5] Terrain grid (upsample {self.upsample_factor}×, bilinear)...")
        try:
            z_grid, meta = self.converter.build_grid(
                tif_path, upsample_factor=self.upsample_factor
            )
        except Exception as e:
            print(f"  ✗ {e}")
            return result
        print(
            f"  ✓ {meta['h']}×{meta['w']} buněk  "
            f"≈ {meta['cell_width_m']:.1f}×{meta['cell_height_m']:.1f} m/buňka"
        )

        # ── 3. Terrain OBJ (always with UV coords for SDF) ───────────────
        print("\n[3/5] Terrain OBJ...")
        try:
            terrain_path = self.converter.write_obj_with_uv(
                z_grid, meta, obj_name=f"{output_base_name}_terrain"
            )
            result["terrain"] = terrain_path
            print(f"  ✓ {terrain_path}")
        except Exception as e:
            print(f"  ✗ {e}")
            return result

        # ── 4. OSM data + SDF texture + Buildings ─────────────────────────
        highways: list = []
        if self.osm_client:
            osm_dir = self.converter.output_dir
            print("\n[4/5] OSM data...")
            highways = self._load_or_fetch(
                bbox,
                cache_path=os.path.join(osm_dir, f"{output_base_name}_osm_highways.json"),
                fetch_fn=self.osm_client.get_highways,
                label="silnic",
            )
            print(f"  ✓ {len(highways)} silnic")

        if self.sdf_generator and highways:
            print("  SDF textura silnic...")
            try:
                sdf_path = os.path.join(
                    self.converter.output_dir, f"{output_base_name}_roads_sdf.png"
                )
                self.sdf_generator.generate(highways, meta, sdf_path)
                result["sdf_texture"] = sdf_path
                print(f"  ✓ {sdf_path}")
            except Exception as e:
                traceback.print_exc()
                print(f"  ✗ SDF selhal: {e}")

        if self.osm_client and self.building_extruder:
            osm_dir = self.converter.output_dir
            _fetch_fn = getattr(
                self.osm_client, "get_buildings_with_parts",
                self.osm_client.get_buildings,
            )
            buildings = self._load_or_fetch(
                bbox,
                cache_path=os.path.join(osm_dir, f"{output_base_name}_osm_buildings.json"),
                fetch_fn=_fetch_fn,
                label="budov",
            )
            if buildings:
                print(f"  Budovy ({len(buildings)})...")
                try:
                    bld_path = self.building_extruder.generate(
                        buildings, z_grid, meta, obj_name=output_base_name
                    )
                    result["buildings"] = bld_path
                    print(f"  ✓ {bld_path}")
                except Exception as e:
                    print(f"  ✗ {e}")

        # ── 5. GLB export ─────────────────────────────────────────────────
        if self.gltf_exporter:
            print("\n[5/5] GLB export...")
            try:
                cx = meta["total_width_m"]  / 2
                cy = meta["total_height_m"] / 2
                exporter_fn = (
                    self.gltf_exporter.export_draco
                    if self.gltf_exporter.use_draco
                    else self.gltf_exporter.export
                )
                glb_path = exporter_fn(
                    result,
                    output_name=output_base_name,
                    tile_center_local=(cx, cy),
                )
                result["glb"] = glb_path
                print(f"  ✓ {glb_path}")
            except Exception as e:
                traceback.print_exc()
                print(f"  ✗ GLB selhal: {e}")

        print(f"\n{'='*60}")
        print("  PIPELINE HOTOV")
        for k, v in result.items():
            print(f"    {k:12s}: {v or '--'}")
        print(f"{'='*60}\n")
        return result

    # ------------------------------------------------------------------
    # OSM cache helpers
    # ------------------------------------------------------------------

    def _load_or_fetch(
        self,
        bbox: tuple,
        cache_path: str,
        fetch_fn,
        label: str,
    ) -> list:
        """
        Return OSM features from a JSON cache file when available, otherwise
        call ``fetch_fn(bbox)``, save the result, and return it.

        The cache is skipped (and overwritten) when ``self.osm_force_download``
        is True.
        """
        if not self.osm_force_download and os.path.exists(cache_path):
            try:
                with open(cache_path, encoding="utf-8") as fh:
                    data = json.load(fh)
                print(f"  ↩ {label} načteno z cache: {cache_path} ({len(data)} prvků)")
                return data
            except Exception as exc:
                print(f"  ⚠ Cache {cache_path} nelze načíst ({exc}), stahuji znovu...")

        data = fetch_fn(bbox)

        if data:
            try:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False)
                print(f"  ✓ {label} uloženo do cache: {cache_path}")
            except Exception as exc:
                print(f"  ⚠ Nepodařilo se uložit cache ({exc})")

        return data


def main():
    import sys
    sys.path.insert(0, ".")

    from src.extraction.opentopography_client import OpenTopographyClient
    from src.extraction.osm_client             import OsmClient
    from src.conversion.terrain_converter      import TerrainConverter
    from src.conversion.road_mesh              import RoadMesh
    from src.conversion.building_extruder      import BuildingExtruder

    output_dir = "outputs/test_results"

    pipeline = TerrainPipeline(
        client            = OpenTopographyClient(),
        converter         = TerrainConverter(output_dir=output_dir),
        road_mesh         = RoadMesh(output_dir=output_dir),
        osm_client        = OsmClient(),
        building_extruder = BuildingExtruder(output_dir=output_dir),
        upsample_factor   = UPSAMPLE_FACTOR,
    )

    test_bbox = (50.07, 14.43, 50.08, 14.44)
    pipeline.run_pipeline(test_bbox, "prague_test_area")


if __name__ == "__main__":
    main()
