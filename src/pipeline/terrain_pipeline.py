"""
Full terrain pipeline
=====================
  1. DEM download (OpenTopography)
  2. Terrain grid  – bilinear upsample, smooth SRTM grid (no CDT distortion)
  3. Terrain OBJ   – regular quad mesh
  4. OSM highways  – download
  5. Road 3D OBJ   – top surface + side walls (always above terrain, no gap)
  6. Buildings OBJ – min-vertex Z base (never floats)

Outputs (all in output_dir):
  <base>_terrain.obj     ← plain smooth terrain
  <base>_roads.obj       ← 3-D road strips with side walls
  <base>_buildings.obj   ← extruded OSM buildings
"""
import os

UPSAMPLE_FACTOR = 10   # 30 m SRTM → ≈ 3 m grid  (try 20 for ≈ 1.5 m)


class TerrainPipeline:

    def __init__(
        self,
        client,
        converter,
        road_mesh,
        osm_client        = None,
        building_extruder = None,
        sdf_generator     = None,
        upsample_factor:    int = UPSAMPLE_FACTOR,
    ):
        self.client            = client
        self.converter         = converter
        self.road_mesh         = road_mesh
        self.osm_client        = osm_client
        self.building_extruder = building_extruder
        self.sdf_generator     = sdf_generator
        self.upsample_factor   = upsample_factor

    def run_pipeline(self, bbox: tuple, output_base_name: str) -> dict:
        print(f"\n{'='*60}")
        print(f"  Pipeline: '{output_base_name}'  bbox={bbox}")
        print(f"  Upsample: {self.upsample_factor}×")
        print(f"{'='*60}")

        result = {
            "terrain": None,
            "roads": None,
            "buildings": None,
            "sdf_texture": None,
        }

        # ── 1. DEM ────────────────────────────────────────────────────────
        print("\n[1/5] Stahuji DEM...")
        tif_path = self.client.get_dem(bbox)
        if not tif_path or not os.path.exists(tif_path):
            print("  ✗ Nepodařilo se stáhnout DEM.")
            return result
        print(f"  ✓ {tif_path}")

        # ── 2. Terrain grid (bilinear upsample) ───────────────────────────
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

        # ── 3. Terrain OBJ ────────────────────────────────────────────────
        print("\n[3/6] Zapisuji terrain OBJ...")
        try:
            use_uv = self.sdf_generator is not None
            writer = (
                self.converter.write_obj_with_uv if use_uv
                else self.converter.write_obj
            )
            terrain_path = writer(
                z_grid, meta, obj_name=f"{output_base_name}_terrain"
            )
            result["terrain"] = terrain_path
            print(f"  ✓ {terrain_path}")
        except Exception as e:
            print(f"  ✗ {e}")
            return result

        # ── 4. OSM highways ──────────────────────────────────────────────
        highways: list = []
        if self.osm_client:
            print("\n[4/6] Stahuji OSM silnice + budovy...")
            highways = self.osm_client.get_highways(bbox)
            print(f"  ✓ {len(highways)} silnic")

        # ── 4b. SDF road texture ─────────────────────────────────────────
        if self.sdf_generator and highways:
            print("\n[4b/6] Generuji SDF texturu silnic...")
            try:
                sdf_path = os.path.join(
                    self.converter.output_dir, f"{output_base_name}_roads_sdf.png"
                )
                self.sdf_generator.generate(highways, meta, sdf_path)
                result["sdf_texture"] = sdf_path
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  ✗ SDF selhal: {e}")

        # ── 5. Road 3D OBJ ──────────────────────────────────────────────
        if self.road_mesh and highways:
            print("\n[5/6] Road 3D OBJ (top + side walls)...")
            try:
                road_path = self.road_mesh.generate_obj(
                    highways, z_grid, meta,
                    obj_name=output_base_name,
                )
                result["roads"] = road_path
                print(f"  ✓ {road_path}")
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  ✗ Road mesh selhal: {e}")

        # ── 6. Buildings ─────────────────────────────────────────────────
        if self.osm_client and self.building_extruder:
            buildings = self.osm_client.get_buildings(bbox)
            if buildings:
                print(f"\n[6] Budovy ({len(buildings)})...")
                try:
                    bld_path = self.building_extruder.extrude_buildings(
                        buildings, z_grid, meta, obj_name=output_base_name
                    )
                    result["buildings"] = bld_path
                    print(f"  ✓ {bld_path}")
                except Exception as e:
                    print(f"  ✗ {e}")

        print(f"\n{'='*60}")
        print("  PIPELINE HOTOV")
        for k, v in result.items():
            print(f"    {k:12s}: {v or '--'}")
        print(f"{'='*60}\n")
        return result


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
