import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import ee
import httpx

app = FastAPI(title="Geospatial Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

GCP_PROJECT = os.getenv("EE_PROJECT", "able-pottery-463117-r0")
DEFAULT_CENTER = [16.506206877709086, 80.648075109326]
DEFAULT_ZOOM = 12

try:
    ee.Initialize(project=GCP_PROJECT)
    EE_READY = True
except Exception as e:
    print(f"Earth Engine initialization warning: {e}")
    EE_READY = False


def get_gee_tile_url(image: ee.Image, vis_params: dict) -> str:
    map_id_dict = image.getMapId(vis_params)
    return map_id_dict["tile_fetcher"].url_format


@app.get("/api/layers")
def get_layers(
    lat: float = Query(DEFAULT_CENTER[0]),
    lon: float = Query(DEFAULT_CENTER[1]),
    radius_m: float = Query(5000.0, ge=250.0, le=50000.0),
    start_date: str = Query("2026-09-01"),
    end_date: str = Query("2026-10-15"),
):
    """
    Returns live Google Earth Engine tile URLs and zonal statistics (NDVI, EVI)
    for a circular buffer of radius_m meters around (lat, lon).
    """
    if not EE_READY:
        raise HTTPException(
            status_code=503,
            detail="Google Earth Engine is not authenticated on the server. Run 'earthengine authenticate'.",
        )

    point = ee.Geometry.Point([lon, lat])
    study_area = point.buffer(radius_m)

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(study_area)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
        .clip(study_area)
    )

    # 1. Sentinel-2 RGB
    rgb_url = get_gee_tile_url(
        s2, {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}
    )

    # 2. NDVI: (NIR - Red) / (NIR + Red)
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndvi_url = get_gee_tile_url(
        ndvi,
        {
            "min": -0.2,
            "max": 0.8,
            "palette": ["#8c510a", "#f6e8c3", "#d9f0d3", "#1b7837"],
        },
    )

    # 3. EVI: 2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))
    evi = s2.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {
            "NIR": s2.select("B8").divide(10000),
            "RED": s2.select("B4").divide(10000),
            "BLUE": s2.select("B2").divide(10000),
        },
    ).rename("EVI")
    evi_url = get_gee_tile_url(
        evi,
        {
            "min": -0.2,
            "max": 0.8,
            "palette": ["#762a83", "#f7f7f7", "#5aae61", "#00441b"],
        },
    )

    # 4. Study Area Boundary (Circular Perimeter)
    boundary_img = ee.Image().byte().paint(
        featureCollection=ee.FeatureCollection([ee.Feature(study_area)]),
        color=1,
        width=2,
    )
    boundary_url = get_gee_tile_url(boundary_img, {"palette": ["#d9534f"]})

    # Compute Zonal Mean Statistics inside study_area
    scale = 30 if radius_m <= 10000 else 100
    stats_image = ndvi.addBands(evi)
    stats_dict = stats_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=study_area,
        scale=scale,
        maxPixels=1e8,
        bestEffort=True,
    ).getInfo()

    ndvi_val = stats_dict.get("NDVI")
    evi_val = stats_dict.get("EVI")

    return {
        "center": [lat, lon],
        "radius_m": radius_m,
        "zoom": DEFAULT_ZOOM,
        "stats": {
            "ndvi_mean": round(ndvi_val, 4) if ndvi_val is not None else None,
            "evi_mean": round(evi_val, 4) if evi_val is not None else None,
        },
        "layers": [
            {"id": "s2_rgb", "name": "Sentinel-2 RGB", "url": rgb_url, "visible": True},
            {"id": "ndvi", "name": "NDVI", "url": ndvi_url, "visible": False},
            {"id": "evi", "name": "EVI", "url": evi_url, "visible": False},
            {
                "id": "boundary",
                "name": "Study area boundary",
                "url": boundary_url,
                "visible": True,
            },
        ],
    }


@app.get("/api/geocode")
async def geocode_location(q: str = Query(..., min_length=2)):
    if "," in q:
        parts = [p.strip() for p in q.split(",")]
        if len(parts) == 2:
            try:
                lat, lon = float(parts[0]), float(parts[1])
                return [{"name": f"Coordinates ({lat}, {lon})", "lat": lat, "lon": lon}]
            except ValueError:
                pass

    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": q, "format": "json", "limit": 5}
    headers = {"User-Agent": "GeospatialMonitorApp/1.0"}

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params, headers=headers, timeout=10.0)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="Geocoding service unavailable")
        data = resp.json()

    return [
        {
            "name": item.get("display_name"),
            "lat": float(item["lat"]),
            "lon": float(item["lon"]),
        }
        for item in data
    ]