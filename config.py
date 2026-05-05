"""
svarog-engine  –  centrální konfigurace
========================================
Edituj tento soubor pro změnu chování pipeline.
Všechny vzdálenosti jsou v metrech, výšky v metrech nad terénem.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SCÉNA – co a kam generovat
# ─────────────────────────────────────────────────────────────────────────────

# Geografická oblast (WGS-84, pořadí: jih, západ, sever, východ)
BBOX = (50.07, 14.43, 50.08, 14.44)

# Výstupní složka a prefix souborů
OUTPUT_DIR  = "outputs/test_results"
OUTPUT_NAME = "prague_test_area"


# ─────────────────────────────────────────────────────────────────────────────
# TERÉN
# ─────────────────────────────────────────────────────────────────────────────

# Koeficient převzorkování SRTM mřížky (bilineární interpolace, bez overshoot).
#   1  = nativní SRTM ~30 m/buňku
#   10 = ~3 m/buňku   →  130 k vrcholů  (doporučeno)
#   20 = ~1.5 m/buňku →  520 k vrcholů
TERRAIN_UPSAMPLE = 10


# ─────────────────────────────────────────────────────────────────────────────
# SILNICE
# ─────────────────────────────────────────────────────────────────────────────

# Sloučit překrývající se pruhy silnic v 2D (Shapely unary_union).
# True = správné chování (výchozí), False = bez sloučení.
ROAD_USE_2D_UNION = True

# Zjednodušení hranice sjednoceného polygonu [m].
ROAD_BOUNDARY_SIMPLIFY = 0.15

# ── Z-burn ────────────────────────────────────────────────────────────────
# Hloubka zahloubení silniční plochy do terénu [m].
# Terrain vrcholy uvnitř road polygonu jsou sníženy o tuto hodnotu.
# Silnice stále kopíruje terén (každý svah/kopec), ale je mírně zapuštěna.
#   0.05 m – jemné, sotva viditelné (plochý terén)
#   0.08 m – doporučeno (výchozí)
#   0.15 m – výrazné zahloubení (členitý terén)
ROAD_BURN_DEPTH = 0.08

# ── Road textura ──────────────────────────────────────────────────────────
# Rozlišení čtvercové UV textury [px].
#   512  → ~2 m/px pro 1 km dlaždici  (doporučeno pro mobil)
#   1024 → ~1 m/px  (vyšší kvalita, ~4× větší soubor)
ROAD_TEXTURE_SIZE = 512

# Poloměr Gaussian blur na hranách silnice [px].
# Větší = měkčí přechod do terénu.
#   0   = tvrdá hrana (žádný blur)
#   4   = doporučeno (přibližně 8 m při 512px na 1 km)
#   8   = velmi měkký přechod
ROAD_TEXTURE_BLUR_PX = 4

# Barvy silnic a terénu v textuře (RGB).
ROAD_COLOR    = (72,  72,  72)   # tmavě šedý asfalt
TERRAIN_COLOR = (160, 148, 124)  # neutrální písčito-hnědá

# OSM typy silnic, které se NEZPRACOVÁVAJÍ (stavba, návrh, zvláštní apod.)
ROAD_EXCLUDED_TYPES = {
    "proposed", "construction", "bus_guideway",
    "escape", "raceway", "corridor", "platform",
}

# Polorozchod silnic podle typu OSM [m] (polovina celkové šíře vozovky).
# Pokud OSM tag 'width' chybí, použije se tato tabulka.
ROAD_HALF_WIDTHS = {
    "motorway":       6.0,  "motorway_link":   3.5,
    "trunk":          5.0,  "trunk_link":       3.0,
    "primary":        4.5,  "primary_link":     2.5,
    "secondary":      3.5,  "secondary_link":   2.0,
    "tertiary":       3.0,  "tertiary_link":    1.8,
    "unclassified":   2.5,
    "residential":    2.5,  "living_street":    2.0,
    "service":        1.8,
    "track":          1.8,
    "cycleway":       1.2,  "pedestrian":       2.5,
    "footway":        1.0,  "path":             0.8,
    "steps":          1.0,
}
ROAD_DEFAULT_HALF_WIDTH = 2.0   # fallback

# Krok subdivize osy silnice [m].
# Menší = hustší vzorkování terénu podél silnice = přesnější kopírování výšky.
# Používá se pouze při ROAD_USE_2D_UNION = False (strip-based fallback).
ROAD_SUBDIV_BY_TYPE = {
    "motorway":     0.5, "motorway_link":  0.5,
    "trunk":        0.5, "trunk_link":     0.5,
    "primary":      0.5, "primary_link":   0.5,
    "secondary":    1.0, "secondary_link": 1.0,
    "tertiary":     1.0, "tertiary_link":  1.0,
    "unclassified": 1.5,
    "residential":  2.0, "living_street":  2.0,
    "service":      2.0, "track":          2.0,
    "cycleway":     2.0, "pedestrian":     2.0,
    "footway":      3.0, "path":           3.0,
    "steps":        3.0,
}
ROAD_DEFAULT_SUBDIV = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# BUDOVY
# ─────────────────────────────────────────────────────────────────────────────

# Výchozí výška patra [m] – použije se pokud OSM neobsahuje 'height' ani 'building:levels'.
BUILDING_HEIGHT_PER_FLOOR = 3.0

# Výchozí počet pater – pokud OSM neobsahuje ani jedno z výše uvedeného.
BUILDING_DEFAULT_FLOORS = 2

# Výchozí celková výška [m] – absolutní fallback.
BUILDING_DEFAULT_HEIGHT = 6.0


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMALIZACE (budoucí – zatím nevyužito, připraveno pro budoucí fázi)
# ─────────────────────────────────────────────────────────────────────────────

# Sloučit vrcholy, které jsou blíže než tato vzdálenost [m].
# 0 = vypnuto.  Doporučeno: 0.01 (1 cm) pro redukci duplikátů.
OPTIMIZE_WELD_THRESHOLD = 0.0

# Zjednodušit terénní mesh (Ramer-Douglas-Peucker tolerance) [m].
# 0 = vypnuto.  Doporučeno: 0.05–0.20 pro LOD export.
OPTIMIZE_TERRAIN_SIMPLIFY = 0.0

# Komprimovat OBJ výstup do .gz.
OPTIMIZE_COMPRESS_OUTPUT = False
