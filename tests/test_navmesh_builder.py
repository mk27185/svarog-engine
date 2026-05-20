from src.conversion.navmesh_builder import NavmeshBuilder


def test_navmesh_full_tile_without_buildings():
    meta = {"total_width_m": 200.0, "total_height_m": 200.0}
    result = NavmeshBuilder().build([], meta)
    assert result is not None
    assert len(result["vertices"]) >= 4
    assert len(result["indices"]) >= 1
    assert result["walkable_area_m2"] > 30_000
