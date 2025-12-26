#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fl-s1-radar (STAC) – VV/VH + Differenz (t0/t-1) + "All-Weather"-Kapitel

Deploy: Google Cloud Run (Source Deploy) – ein einziges Python-Skript.

Empfohlene requirements.txt (Beispiel):
  Flask>=2.3
  requests
  numpy
  rasterio
  shapely
  pyproj
  Pillow

Wichtige ENV-Variablen:
  APP_TITLE                 (default: "FieldLense – Sentinel-1 Radar (STAC)")
  STAC_API                  (default: https://planetarycomputer.microsoft.com/api/stac/v1)
  STAC_COLLECTION           (default: sentinel-1-grd)
  PC_SAS_TOKEN_URL          (default: https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection})
  PC_TOKEN_TTL_SECONDS      (default: 1800)

  DEFAULT_N                 (default: 12)
  MAX_N                     (default: 50)

  MAX_AOI_AREA_KM2          (default: 25.0)
  MAX_RASTER_DIM_PX         (default: 1536)   # Preview/Download sampling cap
  HTTP_TIMEOUT              (default: 60)

  DB_FACTOR                 (default: 10.0)   # 10=Power dB, 20=Amplitude dB

  TMP_DIR                   (default: /tmp)
  CACHE_TTL_SECONDS         (default: 3600)
  MAX_CACHE_ITEMS           (default: 60)
"""

import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from flask import Flask, Response, jsonify, render_template_string, request, send_file
from PIL import Image
from pyproj import Transformer
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform

import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, Window
from rasterio.warp import transform_bounds, reproject
from rasterio.crs import CRS
from affine import Affine
import rasterio.windows
import rasterio.transform

# -------------------------------
# Config
# -------------------------------

APP_TITLE = os.getenv("APP_TITLE", "FieldLense – Sentinel-1 Radar (STAC)")

STAC_API = os.getenv("STAC_API", "https://planetarycomputer.microsoft.com/api/stac/v1").rstrip("/")
STAC_COLLECTION = os.getenv("STAC_COLLECTION", "sentinel-1-grd")

PC_SAS_TOKEN_URL = os.getenv(
    "PC_SAS_TOKEN_URL",
    "https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}",
)

PC_TOKEN_TTL_SECONDS = int(os.getenv("PC_TOKEN_TTL_SECONDS", "1800"))

DEFAULT_N = int(os.getenv("DEFAULT_N", "12"))
MAX_N = int(os.getenv("MAX_N", "50"))

MAX_AOI_AREA_KM2 = float(os.getenv("MAX_AOI_AREA_KM2", "25.0"))
MAX_RASTER_DIM_PX = int(os.getenv("MAX_RASTER_DIM_PX", "1536"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60"))

DB_FACTOR = float(os.getenv("DB_FACTOR", "10.0"))  # 10=power->dB, 20=amplitude->dB

TMP_DIR = Path(os.getenv("TMP_DIR", "/tmp")) / "fl_s1_cache"
TMP_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
MAX_CACHE_ITEMS = int(os.getenv("MAX_CACHE_ITEMS", "60"))

# -------------------------------
# Flask
# -------------------------------

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


# -------------------------------
# Helpers
# -------------------------------

@dataclass
class RenderResult:
    job_id: str
    bounds_wgs84: Tuple[Tuple[float, float], Tuple[float, float]]  # ((south, west), (north, east))
    aoi_area_km2: float
    mode: str
    png_path: Path
    tif_path: Path
    t0: Dict[str, Any]
    t1: Optional[Dict[str, Any]]
    debug: Dict[str, Any]


_token_cache: Dict[str, Any] = {"token": None, "exp": 0}


def _cleanup_cache() -> None:
    """Best-effort cleanup (Cloud Run /tmp)."""
    try:
        items = []
        for p in TMP_DIR.glob("*"):
            if p.is_file():
                items.append((p.stat().st_mtime, p))
        items.sort(reverse=True)  # newest first

        now = time.time()

        # TTL cleanup
        for mtime, p in items:
            if now - mtime > CACHE_TTL_SECONDS:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

        # size cap cleanup
        items = []
        for p in TMP_DIR.glob("*"):
            if p.is_file():
                items.append((p.stat().st_mtime, p))
        items.sort(reverse=True)
        if len(items) > MAX_CACHE_ITEMS:
            for _, p in items[MAX_CACHE_ITEMS:]:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
    except Exception:
        pass


def _parse_geojson(payload: Any) -> Dict[str, Any]:
    if payload is None:
        raise ValueError("Kein GeoJSON übergeben.")
    if isinstance(payload, str):
        payload = payload.strip()
        if not payload:
            raise ValueError("Leerer GeoJSON-String.")
        return json.loads(payload)
    if isinstance(payload, dict):
        return payload
    raise ValueError("GeoJSON muss ein JSON-Objekt oder String sein.")


def _extract_single_geometry(gj: Dict[str, Any]):
    """Accept Feature, FeatureCollection(1), or plain Geometry."""
    t = gj.get("type")
    if t == "Feature":
        geom = gj.get("geometry")
        if not geom:
            raise ValueError("Feature ohne geometry.")
        return shape(geom)
    if t == "FeatureCollection":
        feats = gj.get("features") or []
        if len(feats) != 1:
            raise ValueError("FeatureCollection muss genau 1 Feature enthalten.")
        geom = feats[0].get("geometry")
        if not geom:
            raise ValueError("Feature ohne geometry.")
        return shape(geom)
    if t in ("Polygon", "MultiPolygon"):
        return shape(gj)
    raise ValueError(f"Nicht unterstützter GeoJSON-Typ: {t}. Erlaubt: Feature, FeatureCollection(1), Polygon, MultiPolygon.")


def _transformer(src_epsg: int, dst_epsg: int) -> Transformer:
    return Transformer.from_crs(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True)


def _geom_to_epsg(geom, src_epsg: int, dst_epsg: int):
    tr = _transformer(src_epsg, dst_epsg)
    return shp_transform(lambda x, y: tr.transform(x, y), geom)


def _utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    if lat >= 0:
        return 32600 + zone
    return 32700 + zone


def _bounds_epsg_to_wgs84(minx: float, miny: float, maxx: float, maxy: float, src_epsg: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    tr = _transformer(src_epsg, 4326)
    west, south = tr.transform(minx, miny)
    east, north = tr.transform(maxx, maxy)
    return (south, west), (north, east)


def _compute_scaled_dims(w_px: float, h_px: float, max_dim_px: int) -> Tuple[int, int]:
    if w_px <= 0 or h_px <= 0:
        raise ValueError("Ungültige Ausdehnung (width/height <= 0).")
    if w_px >= h_px:
        out_w = max_dim_px
        out_h = max(1, int(round(max_dim_px * (h_px / w_px))))
    else:
        out_h = max_dim_px
        out_w = max(1, int(round(max_dim_px * (w_px / h_px))))
    return out_w, out_h


def _pc_get_sas_token(collection_id: str) -> str:
    """Fetch (and cache) Planetary Computer SAS token for the given collection."""
    now = time.time()
    tok = _token_cache.get("token")
    exp = float(_token_cache.get("exp") or 0)
    if tok and now < exp:
        return tok

    url = PC_SAS_TOKEN_URL.format(collection=collection_id)
    r = requests.get(url, timeout=HTTP_TIMEOUT)
    if not r.ok:
        raise ValueError(f"SAS-Token Request fehlgeschlagen ({r.status_code}). Antwort: {r.text[:800]}")
    data = r.json()
    token = data.get("token")
    if not token:
        raise ValueError("SAS-Token Antwort enthält kein 'token' Feld.")
    _token_cache["token"] = token
    _token_cache["exp"] = now + max(60, min(PC_TOKEN_TTL_SECONDS, 3600))
    return token


def _sign_href_if_needed(href: str) -> str:
    """Append SAS token for Planetary Computer Azure Blob URLs (best effort)."""
    # If user points STAC_API elsewhere, this still doesn't break – it will just append token only if it looks like PC.
    if "planetarycomputer.microsoft.com" not in STAC_API and "blob.core.windows.net" not in href:
        return href

    # Already signed?
    if "sig=" in href or "se=" in href:
        return href

    token = _pc_get_sas_token(STAC_COLLECTION)
    if "?" in href:
        return href + "&" + token
    return href + "?" + token


def _stac_search(intersects: Dict[str, Any], n: int, orbit_state: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Return STAC items (raw dicts), sorted desc by datetime.
    No sortby usage (avoids upstream sort errors).
    """
    n = max(1, min(int(n), MAX_N))
    # we fetch more than N to be able to find a stable t0/t-1 pair (same orbit/epsg)
    limit = min(max(n * 6, 30), 200)

    url = f"{STAC_API}/search"
    body: Dict[str, Any] = {
        "collections": [STAC_COLLECTION],
        "intersects": intersects,
        "limit": limit,
    }

    # Keep datetime open-ended; user can add later.
    # We avoid server-side "sortby" entirely.
    r = requests.post(url, json=body, timeout=HTTP_TIMEOUT)
    if not r.ok:
        # Try JSON first, else plain
        try:
            j = r.json()
            raise ValueError(f"STAC Request fehlgeschlagen: Upstream HTTP {r.status_code}: {j}")
        except Exception:
            raise ValueError(f"STAC Request fehlgeschlagen: Upstream HTTP {r.status_code}: {r.text[:1200]}")

    data = r.json()
    feats = data.get("features") or []
    if not feats:
        return [], {"stac_url": url, "stac_body": body, "count": 0}

    def _dt(item: Dict[str, Any]) -> str:
        props = item.get("properties") or {}
        return props.get("datetime") or props.get("start_datetime") or ""

    feats.sort(key=_dt, reverse=True)

    # Optional filter by orbit_state (ascending/descending)
    if orbit_state and orbit_state.lower() in ("ascending", "descending"):
        want = orbit_state.lower()
        out = []
        for it in feats:
            props = it.get("properties") or {}
            v = (props.get("sat:orbit_state") or props.get("orbit_state") or "").lower()
            if v == want:
                out.append(it)
        feats = out

    return feats, {"stac_url": url, "stac_body": body, "count": len(feats)}

def _stac_proj_info(item: dict, asset: dict):
    """
    Liefert (crs, transform_affine, shape_yx) aus STAC Projection Extension.
    Asset-Level überschreibt Item-Level.
    """
    props = item.get("properties", {}) or {}

    # Projection extension (neu: proj:code; alt: proj:epsg)
    code = asset.get("proj:code") or props.get("proj:code")
    epsg = asset.get("proj:epsg") or props.get("proj:epsg")
    wkt2 = asset.get("proj:wkt2") or props.get("proj:wkt2")

    transform = asset.get("proj:transform") or props.get("proj:transform")
    shape = asset.get("proj:shape") or props.get("proj:shape")

    crs = None
    if isinstance(code, str) and code.strip():
        # z.B. "EPSG:4326" / "EPSG:32632"
        crs = CRS.from_string(code.strip())
    elif epsg is not None:
        crs = CRS.from_epsg(int(epsg))
    elif isinstance(wkt2, str) and wkt2.strip():
        crs = CRS.from_wkt(wkt2)

    aff = None
    if isinstance(transform, (list, tuple)):
        # STAC proj:transform ist im Rasterio-Order (a,b,c,d,e,f). :contentReference[oaicite:2]{index=2}
        if len(transform) == 6:
            a, b, c, d, e, f = transform
            aff = Affine(a, b, c, d, e, f)
        elif len(transform) == 9:
            # 3x3 row-major; erste 2 Zeilen reichen
            a, b, c, d, e, f = transform[:6]
            aff = Affine(a, b, c, d, e, f)

    shape_yx = None
    if isinstance(shape, (list, tuple)) and len(shape) == 2:
        # proj:shape ist (Y, X). :contentReference[oaicite:3]{index=3}
        shape_yx = (int(shape[0]), int(shape[1]))

    return crs, aff, shape_yx

def _asset_candidates(item: Dict[str, Any], pol: str) -> List[Dict[str, Any]]:
    assets = item.get("assets") or {}
    pol_low = pol.lower()

    def _is_previewish(k: str, a: Dict[str, Any]) -> bool:
        kl = k.lower()
        roles = [str(x).lower() for x in (a.get("roles") or [])]
        title = str(a.get("title") or "").lower()
        return (
            "thumbnail" in kl or "preview" in kl or "rendered" in kl or "tilejson" in kl
            or "thumbnail" in roles or "overview" in roles
            or "preview" in title or "thumbnail" in title
        )

    def _is_tiffish(href: str, a: Dict[str, Any]) -> bool:
        typ = str(a.get("type") or "").lower()
        h = href.lower()
        return ("tiff" in typ) or h.endswith(".tif") or h.endswith(".tiff") or ("geotiff" in typ)

    cands: List[Dict[str, Any]] = []
    for k, a in assets.items():
        if not isinstance(a, dict):
            continue
        href = a.get("href")
        if not href:
            continue

        if _is_previewish(str(k), a):
            continue
        if not _is_tiffish(str(href), a):
            continue

        kl = str(k).lower()
        roles = [str(x).lower() for x in (a.get("roles") or [])]
        typ = str(a.get("type") or "").lower()

        score = 0
        if kl == pol_low:
            score += 100
        if pol_low in kl:
            score += 20
        if "data" in roles:
            score += 10
        if "geotiff" in typ:
            score += 5

        cands.append({
            "key": str(k),
            "href": str(href),
            "type": str(a.get("type") or ""),
            "roles": roles,
            "title": str(a.get("title") or ""),
            "score": score,
        })

    cands.sort(key=lambda d: d["score"], reverse=True)
    return cands


def _pick_asset(item: Dict[str, Any], pol: str) -> Dict[str, Any]:
    cands = _asset_candidates(item, pol)
    if not cands:
        assets = list((item.get("assets") or {}).keys())
        raise ValueError(f"Kein geeignetes GeoTIFF-Asset für '{pol}' gefunden. Assets: {assets[:40]}")
    return cands[0]


def _item_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    props = item.get("properties") or {}
    summ = {
        "id": item.get("id"),
        "datetime": props.get("datetime") or props.get("start_datetime"),
        "platform": props.get("platform"),
        "orbit_state": props.get("sat:orbit_state") or props.get("orbit_state"),
        "relative_orbit": props.get("sat:relative_orbit") or props.get("s1:relative_orbit"),
        "epsg": props.get("proj:epsg"),
        "polarizations": props.get("sar:polarizations") or props.get("polarizations"),
    }
    return summ


def _select_t0_t1(items: List[Dict[str, Any]], pol: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Choose t0 = newest item with requested polarization.
    Choose t-1 = next older item matching (orbit_state, relative_orbit, epsg) if possible.
    """
    if not items:
        raise ValueError("Keine Items gefunden (für diese AOI / Filter).")

    # keep only items that actually provide the requested pol asset
    usable = []
    for it in items:
        try:
            _pick_asset(it, pol)
            usable.append(it)
        except Exception:
            continue
    if not usable:
        raise ValueError(f"Keine Items gefunden, die Polarisation '{pol}' enthalten.")

    t0 = usable[0]
    p0 = t0.get("properties") or {}
    want_orbit_state = (p0.get("sat:orbit_state") or p0.get("orbit_state") or "").lower()
    want_rel = p0.get("sat:relative_orbit") or p0.get("s1:relative_orbit")
    want_epsg = p0.get("proj:epsg")

    t1 = None
    for cand in usable[1:]:
        pc = cand.get("properties") or {}
        ok = True

        if want_epsg and pc.get("proj:epsg") and pc.get("proj:epsg") != want_epsg:
            ok = False

        if want_orbit_state:
            v = (pc.get("sat:orbit_state") or pc.get("orbit_state") or "").lower()
            if v and v != want_orbit_state:
                ok = False

        if want_rel and (pc.get("sat:relative_orbit") or pc.get("s1:relative_orbit")):
            if (pc.get("sat:relative_orbit") or pc.get("s1:relative_orbit")) != want_rel:
                ok = False

        if ok:
            t1 = cand
            break

    # If strict match failed, relax to same EPSG (if known)
    if t1 is None and want_epsg:
        for cand in usable[1:]:
            pc = cand.get("properties") or {}
            if pc.get("proj:epsg") == want_epsg:
                t1 = cand
                break

    # If still none, just take the next older
    if t1 is None and len(usable) >= 2:
        t1 = usable[1]

    return t0, t1


def _array_to_rgba_png(arr: np.ndarray, kind: str) -> Image.Image:
    """
    Convert float array (with NaN) to RGBA PNG:
      - kind="abs" (VV/VH): percentile stretch
      - kind="diff": symmetric stretch around 0
    """
    if arr.ndim != 2:
        raise ValueError("Unerwartete Raster-Dimensionen (erwarte 2D).")

    finite = np.isfinite(arr)
    if not np.any(finite):
        raise ValueError("Keine gültigen Pixel (AOI evtl. außerhalb der Szene).")

    a = arr[finite].astype(np.float32)

    if kind == "diff":
        # symmetric stretch
        p = np.percentile(np.abs(a), 98)
        p = float(p) if np.isfinite(p) and p > 0 else float(np.max(np.abs(a)))
        if not np.isfinite(p) or p <= 0:
            p = 1.0
        x = np.clip(arr, -p, p)
        scaled = ((x + p) / (2 * p) * 255.0).astype(np.float32)
    else:
        lo = float(np.percentile(a, 2))
        hi = float(np.percentile(a, 98))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(a)), float(np.max(a))
            if hi <= lo:
                hi = lo + 1.0
        x = np.clip(arr, lo, hi)
        scaled = ((x - lo) / (hi - lo) * 255.0).astype(np.float32)

    gray = np.zeros(arr.shape, dtype=np.uint8)
    gray[finite] = np.clip(scaled[finite], 0, 255).astype(np.uint8)

    alpha = np.zeros(arr.shape, dtype=np.uint8)
    alpha[finite] = 255

    rgba = np.dstack([gray, gray, gray, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def _write_geotiff_float32(path: Path, arr: np.ndarray, crs, transform) -> None:
    nodata = -9999.0
    out = arr.astype(np.float32)
    out_filled = np.where(np.isfinite(out), out, nodata).astype(np.float32)

    profile = {
        "driver": "GTiff",
        "height": out.shape[0],
        "width": out.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
        "interleave": "band",
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out_filled, 1)


def _read_pol_db_on_grid(
    item: Dict[str, Any],
    pol: str,
    aoi_geom_wgs84,
    dst_crs=None,
    dst_transform=None,
    dst_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, Any, Any, Tuple[float, float, float, float]]:
    """
    Read polarization raster as dB array on either its own grid or a provided dst grid.
    Robust against assets where CRS/transform are not in the GeoTIFF header but in STAC proj:* metadata.

    Returns: (arr_db, crs, transform, bounds_in_crs)
    """
    from rasterio.crs import CRS

    pol = (pol or "").lower().strip()
    assets = item.get("assets") or {}
    props = item.get("properties") or {}

    def _is_previewish(k: str, a: Dict[str, Any]) -> bool:
        kl = k.lower()
        roles = [str(x).lower() for x in (a.get("roles") or [])]
        title = str(a.get("title") or "").lower()
        return (
            "thumbnail" in kl
            or "preview" in kl
            or "rendered" in kl
            or "tilejson" in kl
            or "thumbnail" in roles
            or "overview" in roles
            or "preview" in title
            or "thumbnail" in title
        )

    def _is_tiffish(href: str, a: Dict[str, Any]) -> bool:
        typ = str(a.get("type") or "").lower()
        h = href.lower()
        return ("tiff" in typ) or h.endswith(".tif") or h.endswith(".tiff") or ("geotiff" in typ)

    def _stac_proj_info(asset_dict: Dict[str, Any]) -> Tuple[Optional[CRS], Optional[rasterio.Affine], Optional[Tuple[int, int]]]:
        # Asset-level overrides item-level
        code = asset_dict.get("proj:code") or props.get("proj:code")
        epsg = asset_dict.get("proj:epsg") or props.get("proj:epsg")
        wkt2 = asset_dict.get("proj:wkt2") or props.get("proj:wkt2")
        tr = asset_dict.get("proj:transform") or props.get("proj:transform")
        sh = asset_dict.get("proj:shape") or props.get("proj:shape")

        crs = None
        if isinstance(code, str) and code.strip():
            crs = CRS.from_string(code.strip())
        elif epsg is not None:
            try:
                crs = CRS.from_epsg(int(epsg))
            except Exception:
                crs = None
        elif isinstance(wkt2, str) and wkt2.strip():
            try:
                crs = CRS.from_wkt(wkt2)
            except Exception:
                crs = None

        aff = None
        if isinstance(tr, (list, tuple)):
            if len(tr) == 6:
                a, b, c, d, e, f = tr
                aff = rasterio.Affine(a, b, c, d, e, f)
            elif len(tr) == 9:
                a, b, c, d, e, f = tr[:6]
                aff = rasterio.Affine(a, b, c, d, e, f)

        shape_yx = None
        if isinstance(sh, (list, tuple)) and len(sh) == 2:
            # STAC proj:shape is (Y, X)
            try:
                shape_yx = (int(sh[0]), int(sh[1]))
            except Exception:
                shape_yx = None

        return crs, aff, shape_yx

    # Build candidates (prefer GeoTIFF data, avoid previews)
    candidates = []
    for k, a in assets.items():
        if not isinstance(a, dict):
            continue
        href = a.get("href")
        if not href:
            continue
        if _is_previewish(str(k), a):
            continue
        if not _is_tiffish(str(href), a):
            continue

        kl = str(k).lower()
        roles = [str(x).lower() for x in (a.get("roles") or [])]
        typ = str(a.get("type") or "").lower()

        score = 0
        if kl == pol:
            score += 100
        if pol in kl:
            score += 20
        if "data" in roles:
            score += 10
        if "geotiff" in typ:
            score += 5

        candidates.append((score, str(k)))

    candidates.sort(reverse=True, key=lambda x: x[0])

    # If we found nothing with the strict filter, fall back to exact key match (still safer than substring).
    if not candidates and pol in assets and isinstance(assets[pol], dict) and assets[pol].get("href"):
        candidates = [(999, pol)]

    if not candidates:
        raise ValueError(f"Item hat kein geeignetes GeoTIFF-Asset für Polarisation '{pol}'. Assets: {list(assets.keys())[:40]}")

    last_err = None

    # AOI bounds in WGS84 (lon/lat)
    b_wgs = aoi_geom_wgs84.bounds  # (minx,miny,maxx,maxy) lon/lat

    for _, asset_key in candidates:
        asset = assets.get(asset_key) or {}
        href = str(asset.get("href") or "")
        if not href:
            continue
        href = _sign_href_if_needed(href)

        try:
            with rasterio.Env(
                GDAL_DISABLE_READDIR_ON_OPEN="YES",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
            ):
                with rasterio.open(href) as src:
                    # Determine CRS/transform: prefer file header, else STAC proj:*
                    crs = src.crs
                    transform = src.transform

                    if crs is None or transform is None or transform == rasterio.Affine.identity():
                        stac_crs, stac_transform, stac_shape = _stac_proj_info(asset)
                        if stac_crs is None or stac_transform is None:
                            raise ValueError(
                                f"Asset '{asset_key}' hat keine Geo-Referenz im File und keine proj:* Metadaten (proj:code/epsg + proj:transform)."
                            )
                        crs = stac_crs
                        transform = stac_transform
                        # Optional sanity check: if shape is given and differs strongly, keep going anyway.
                        # (Catalogs can be inconsistent; not fatal for windowing.)

                    # Transform AOI bounds into raster CRS
                    bb = transform_bounds("EPSG:4326", crs, b_wgs[0], b_wgs[1], b_wgs[2], b_wgs[3], densify_pts=21)
                    left, bottom, right, top = bb

                    # Normalize bounds
                    if right < left:
                        left, right = right, left
                    if top < bottom:
                        bottom, top = top, bottom

                    # Build window (robust)
                    try:
                        w = from_bounds(left, bottom, right, top, transform=transform)
                    except Exception:
                        inv = ~transform
                        c0, r0 = inv * (left, top)
                        c1, r1 = inv * (right, bottom)
                        rmin, rmax = sorted([int(math.floor(r0)), int(math.ceil(r1))])
                        cmin, cmax = sorted([int(math.floor(c0)), int(math.ceil(c1))])
                        w = Window(col_off=cmin, row_off=rmin, width=max(0, cmax - cmin), height=max(0, rmax - rmin))

                    # Round/clip and check empty
                    w = w.round_offsets().round_lengths()
                    w = w.intersection(Window(0, 0, src.width, src.height))
                    if w.width <= 0 or w.height <= 0:
                        raise ValueError("AOI liegt außerhalb der Szene (Window leer).")

                    # Output shape
                    if dst_shape is None:
                        out_w, out_h = _compute_scaled_dims(float(w.width), float(w.height), MAX_RASTER_DIM_PX)
                        dst_h, dst_w = out_h, out_w
                    else:
                        dst_h, dst_w = int(dst_shape[0]), int(dst_shape[1])

                    # Window transform in CRS
                    win_transform = rasterio.windows.transform(w, transform)
                    scale_x = float(w.width) / float(dst_w)
                    scale_y = float(w.height) / float(dst_h)
                    out_transform = win_transform * rasterio.Affine.scale(scale_x, scale_y)

                    # Read data
                    arr = src.read(
                        1,
                        window=w,
                        out_shape=(dst_h, dst_w),
                        resampling=Resampling.bilinear,
                    ).astype(np.float32)

                    # linear -> dB (avoid log(<=0))
                    arr[arr <= 0] = np.nan
                    arr_db = DB_FACTOR * np.log10(arr)

                    # Mask outside AOI
                    try:
                        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
                        aoi_proj = shp_transform(lambda x, y: tr.transform(x, y), aoi_geom_wgs84)
                        m = geometry_mask(
                            [mapping(aoi_proj)],
                            out_shape=(dst_h, dst_w),
                            transform=out_transform,
                            invert=True,
                            all_touched=False,
                        )
                        arr_db[~m] = np.nan
                    except Exception:
                        # best effort: keep unmasked if geometry transform fails
                        pass

                    # Reproject onto dst grid (for diff)
                    if dst_crs is not None and dst_transform is not None and dst_shape is not None:
                        dst = np.full(dst_shape, np.nan, dtype=np.float32)
                        reproject(
                            source=arr_db,
                            destination=dst,
                            src_transform=out_transform,
                            src_crs=crs,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.bilinear,
                            src_nodata=np.nan,
                            dst_nodata=np.nan,
                        )
                        arr_db = dst
                        out_transform = dst_transform
                        crs = dst_crs

                    # Bounds in output CRS
                    h, w2 = arr_db.shape
                    left2, bottom2, right2, top2 = rasterio.transform.array_bounds(h, w2, out_transform)

                    return arr_db, crs, out_transform, (left2, bottom2, right2, top2)

        except Exception as e:
            last_err = e
            continue

    raise ValueError(
        f"Kein brauchbares georeferenziertes '{pol}'-Asset gefunden. Letzter Fehler: {last_err}. "
        f"Kandidaten: {[k for _, k in candidates[:10]]}"
    )


def _render_from_geojson(geojson: Dict[str, Any], n: int, orbit_state: str, mode: str) -> RenderResult:
    _cleanup_cache()

    geom_wgs84 = _extract_single_geometry(geojson)
    if geom_wgs84.is_empty:
        raise ValueError("Geometrie ist leer.")
    if geom_wgs84.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"Nur Polygon/MultiPolygon erlaubt (bekommen: {geom_wgs84.geom_type}).")

    # AOI area check in suitable UTM
    c = geom_wgs84.centroid
    utm = _utm_epsg_for_lonlat(float(c.x), float(c.y))
    geom_utm = _geom_to_epsg(geom_wgs84, 4326, utm)
    aoi_area_km2 = float(geom_utm.area) / 1_000_000.0
    if MAX_AOI_AREA_KM2 > 0 and aoi_area_km2 > MAX_AOI_AREA_KM2:
        raise ValueError(f"AOI ist zu groß: {aoi_area_km2:.3f} km² (Limit: {MAX_AOI_AREA_KM2:.3f} km²).")

    mode = (mode or "vv").lower().strip()
    if mode not in ("vv", "vh", "diff_vv", "diff_vh"):
        raise ValueError("Ungültiger mode. Erlaubt: vv, vh, diff_vv, diff_vh")

    pol = "vv" if mode in ("vv", "diff_vv") else "vh"

    # STAC search
    items, stac_dbg = _stac_search(mapping(geom_wgs84), n=n, orbit_state=orbit_state)

    if not items:
        raise ValueError("Keine STAC Items gefunden (Filter/AOI zu strikt?).")

    t0, t1 = _select_t0_t1(items, pol=pol)

    # Read t0
    arr0, crs0, tr0, b0 = _read_pol_db_on_grid(t0, pol=pol, aoi_geom_wgs84=geom_wgs84)

    arr = arr0
    t1_used = None

    if mode.startswith("diff_"):
        if not t1:
            raise ValueError("Für Differenz wird ein t-1 Item benötigt, aber keines gefunden.")
        # Align t1 onto t0 grid
        t1_used = t1
        dst_shape = arr0.shape
        arr1, _, _, _ = _read_pol_db_on_grid(
            t1,
            pol=pol,
            aoi_geom_wgs84=geom_wgs84,
            dst_crs=crs0,
            dst_transform=tr0,
            dst_shape=dst_shape,
        )
        arr = arr0 - arr1
        # require pixels where both valid
        arr[~(np.isfinite(arr0) & np.isfinite(arr1))] = np.nan

    # Create output files
    job_id = uuid.uuid4().hex[:12]
    out_png = TMP_DIR / f"{job_id}.overlay.png"
    out_tif = TMP_DIR / f"{job_id}.data.tif"

    kind = "diff" if mode.startswith("diff_") else "abs"
    img = _array_to_rgba_png(arr, kind=kind)
    img.save(out_png)

    _write_geotiff_float32(out_tif, arr, crs0, tr0)

    # bounds for leaflet overlay in WGS84
    left, bottom, right, top = b0
    try:
        wb = transform_bounds(crs0, "EPSG:4326", left, bottom, right, top, densify_pts=21)
        # Leaflet expects [[south, west],[north, east]]
        bounds_wgs84 = ((wb[1], wb[0]), (wb[3], wb[2]))
    except Exception:
        # fallback using numeric epsg if possible
        epsg = crs0.to_epsg() if crs0 else 4326
        bounds_wgs84 = _bounds_epsg_to_wgs84(left, bottom, right, top, int(epsg or 4326))

    dbg = {
        "pol": pol,
        "db_factor": DB_FACTOR,
        "t0_asset": _sign_href_if_needed((_pick_asset(t0, pol) or {}).get("href", "")),
        "t1_asset": _sign_href_if_needed((_pick_asset(t1_used, pol) or {}).get("href", "")) if t1_used else None,
    }

    return RenderResult(
        job_id=job_id,
        bounds_wgs84=bounds_wgs84,
        aoi_area_km2=aoi_area_km2,
        mode=mode,
        png_path=out_png,
        tif_path=out_tif,
        t0=_item_summary(t0),
        t1=_item_summary(t1_used) if t1_used else None,
        debug=dbg,
    )


# -------------------------------
# Routes + UI
# -------------------------------

INDEX_HTML = """
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{ title }}</title>

  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css" />

  <style>
    :root{
      --bg:#0b0f19;
      --card:#111a2e;
      --text:#e6eaf2;
      --muted:#a8b3cf;
      --border: rgba(255,255,255,.10);
      --primary:#6ea8fe;
      --focus: rgba(110,168,254,.45);
      --radius: 16px;
      --container: 1200px;
      --gap: 14px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --font: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    }
    body{ margin:0; font-family: var(--font); background: var(--bg); color: var(--text); }
    .wrap{
      max-width: var(--container);
      margin: 18px auto;
      padding: 0 14px 24px;
      display: grid;
      grid-template-columns: 1.2fr .8fr;
      gap: var(--gap);
    }
    header{
      max-width: var(--container);
      margin: 18px auto 0;
      padding: 0 14px;
      display:flex;
      align-items:baseline;
      justify-content:space-between;
      gap: 12px;
    }
    h1{ font-size: 18px; margin:0; letter-spacing: .2px; }
    .hint{ color: var(--muted); font-size: 13px; margin-top: 6px; line-height:1.35; }
    .card{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: 0 18px 60px rgba(0,0,0,.35);
      overflow: hidden;
    }
    #map{ height: 70vh; min-height: 520px; }
    .panel{ padding: 12px; display:flex; flex-direction:column; gap: 10px; }
    label{ color: var(--muted); font-size: 12px; }
    textarea{
      width: 100%;
      min-height: 210px;
      resize: vertical;
      background: rgba(255,255,255,.04);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px;
      color: var(--text);
      font-family: var(--mono);
      font-size: 12px;
      outline: none;
    }
    textarea:focus{ border-color: var(--primary); box-shadow: 0 0 0 4px var(--focus); }
    .row{ display:flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    button, select, input{
      appearance:none;
      border: 1px solid var(--border);
      background: rgba(255,255,255,.06);
      color: var(--text);
      padding: 10px 12px;
      border-radius: 12px;
      cursor: pointer;
      font-weight: 600;
    }
    select, input{ font-weight: 500; cursor: default; }
    button.primary{ border-color: rgba(110,168,254,.35); background: rgba(110,168,254,.16); cursor:pointer; }
    button:disabled{ opacity:.55; cursor:not-allowed; }
    .status{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
      padding: 8px 10px;
      border-radius: 12px;
      background: rgba(0,0,0,.18);
      border: 1px solid var(--border);
    }
    .status b{ color: var(--text); }
    .err{ border-color: rgba(255,100,100,.35); background: rgba(255,100,100,.10); color: #ffd1d1; }
    .ok{ border-color: rgba(120,220,160,.35); background: rgba(120,220,160,.08); }
    .small{ font-size: 12px; color: var(--muted); }
    .kpi{
      font-family: var(--mono);
      font-size: 12px;
      color: var(--muted);
      padding: 8px 10px;
      border-radius: 12px;
      background: rgba(255,255,255,.04);
      border: 1px solid var(--border);
      overflow:auto;
      max-height: 180px;
      white-space: pre-wrap;
    }
    details{ border: 1px solid var(--border); border-radius: 12px; padding: 10px 12px; background: rgba(0,0,0,.18); }
    summary{ cursor:pointer; color: var(--text); font-weight: 700; }
    a{ color: var(--primary); text-decoration: none; }
    a:hover{ text-decoration: underline; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>{{ title }}</h1>
      <div class="hint">
        Zeichne ein Polygon oder Rechteck. Es ist immer nur <b>ein</b> Feature aktiv – neues Feature ersetzt das alte.<br>
        Radar (Sentinel-1) ist unabhängig von Wolken/Beleuchtung. Preview: <b>VV/VH</b> oder <b>Δ(t0−t-1)</b> als dB-Overlay.
      </div>
    </div>
    <div class="small">
      API: <code>/api/preview</code>, <code>/api/search</code> · Downloads: <code>/r/&lt;job&gt;/overlay.png</code>, <code>/r/&lt;job&gt;/data.tif</code>
    </div>
  </header>

  <div class="wrap">
    <div class="card">
      <div id="map"></div>
    </div>

    <div class="card">
      <div class="panel">
        <div class="row">
          <button id="btn-clear">AOI löschen</button>
          <button class="primary" id="btn-preview" disabled>Preview Overlay</button>
        </div>

        <div class="row">
          <label>N (letzte Szenen):
            <input id="n" type="number" min="1" max="{{ max_n }}" step="1" value="{{ default_n }}" style="width:90px; margin-left:8px;">
          </label>

          <label>Orbit:
            <select id="orbit" style="margin-left:8px;">
              <option value="">egal</option>
              <option value="ascending">ascending</option>
              <option value="descending">descending</option>
            </select>
          </label>

          <label>Mode:
            <select id="mode" style="margin-left:8px;">
              <option value="vv">VV (t0)</option>
              <option value="vh">VH (t0)</option>
              <option value="diff_vv">ΔVV (t0 − t-1)</option>
              <option value="diff_vh">ΔVH (t0 − t-1)</option>
            </select>
          </label>
        </div>

        <div id="status" class="status">Noch keine AOI.</div>

        <div class="row">
          <button id="btn-geojson" disabled>GeoJSON herunterladen</button>
          <button id="btn-png" disabled>PNG herunterladen</button>
          <button id="btn-tif" disabled>GeoTIFF herunterladen</button>
        </div>

        <label>GeoJSON (aktuelles Feature, EPSG:4326)</label>
        <textarea id="geojson" spellcheck="false" placeholder="Hier erscheint das GeoJSON…"></textarea>

        <details>
          <summary>All-Weather – warum Radar „immer geht“ (Kurzkapitel)</summary>
          <div class="small" style="margin-top:8px; line-height:1.4;">
            <b>Sentinel-1</b> ist ein <b>C-Band SAR</b> (Synthetic Aperture Radar). Es sendet Mikrowellen aktiv aus und misst das zurückgestreute Signal.
            Dadurch ist es weitgehend <b>unabhängig von Wolken</b> und funktioniert auch bei Nacht (keine Sonnenbeleuchtung nötig).
            <br><br>
            <b>VV</b> (co-pol) reagiert u. a. stark auf Oberflächenrauigkeit/Feuchte; <b>VH</b> (cross-pol) ist oft sensibler für Vegetationsstruktur.
            Unterschiedskarten <b>Δ(t0−t-1)</b> sind hilfreich für Change-Detection (z. B. Bodenbearbeitung, Überschwemmung, Ernte, Schneefall, etc.).
            <br><br>
            Hinweis: GRD hat <b>Speckle</b> (körnige Textur). Für Analysen nutzt man oft Filter/Multitemporal-Mittel.
          </div>
        </details>

        <div class="small">
          Server-Limits: <b>{{ max_area_km2 }} km²</b> · Max Preview/Download Dimension: <b>{{ max_dim }} px</b> · dB-Faktor: <b>{{ db_factor }}</b>
        </div>

        <div class="kpi" id="meta" style="display:none;"></div>
      </div>
    </div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>

  <script>
    const map = L.map('map', { preferCanvas: true }).setView([50.1, 8.7], 8);

    const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 20,
      attribution: '&copy; OpenStreetMap'
    }).addTo(map);

    const drawn = new L.FeatureGroup().addTo(map);

    const drawControl = new L.Control.Draw({
      draw: {
        polyline: false,
        circle: false,
        circlemarker: false,
        marker: false,
        polygon: { allowIntersection: false, showArea: true },
        rectangle: true
      },
      edit: {
        featureGroup: drawn,
        edit: true,
        remove: false
      }
    });
    map.addControl(drawControl);

    let overlay = null;
    let currentFeature = null;
    let currentJob = null;

    const elGeo = document.getElementById('geojson');
    const elStatus = document.getElementById('status');
    const elMeta = document.getElementById('meta');
    const btnPreview = document.getElementById('btn-preview');
    const btnClear = document.getElementById('btn-clear');
    const btnGJ = document.getElementById('btn-geojson');
    const btnPNG = document.getElementById('btn-png');
    const btnTIF = document.getElementById('btn-tif');

    const elN = document.getElementById('n');
    const elOrbit = document.getElementById('orbit');
    const elMode = document.getElementById('mode');

    function setStatus(html, cls){
      elStatus.className = 'status' + (cls ? (' ' + cls) : '');
      elStatus.innerHTML = html;
    }

    function setButtons(hasFeature, hasJob){
      btnPreview.disabled = !hasFeature;
      btnGJ.disabled = !hasFeature;
      btnPNG.disabled = !hasJob;
      btnTIF.disabled = !hasJob;
    }

    function clearAll(){
      drawn.clearLayers();
      currentFeature = null;
      currentJob = null;
      elGeo.value = '';
      elMeta.style.display = 'none';
      elMeta.textContent = '';
      if(overlay){
        map.removeLayer(overlay);
        overlay = null;
      }
      setButtons(false, false);
      setStatus('Noch keine AOI.', '');
    }

    function featureToGeoJSON(layer){
      return {
        type: "Feature",
        properties: { epsg: 4326 },
        geometry: layer.toGeoJSON().geometry
      };
    }

    function downloadText(filename, text){
      const blob = new Blob([text], {type: 'application/json;charset=utf-8'});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }

    map.on(L.Draw.Event.CREATED, function (e) {
      drawn.clearLayers();
      if(overlay){
        map.removeLayer(overlay);
        overlay = null;
      }
      currentJob = null;

      const layer = e.layer;
      drawn.addLayer(layer);
      currentFeature = layer;

      const gj = featureToGeoJSON(layer);
      elGeo.value = JSON.stringify(gj, null, 2);

      setButtons(true, false);
      setStatus('AOI gesetzt. Klicke <b>Preview Overlay</b>.', 'ok');
    });

    map.on('draw:edited', function(){
      const layers = drawn.getLayers();
      if(layers.length < 1) return;
      currentFeature = layers[0];
      const gj = featureToGeoJSON(currentFeature);
      elGeo.value = JSON.stringify(gj, null, 2);
      currentJob = null;
      if(overlay){
        map.removeLayer(overlay);
        overlay = null;
      }
      setButtons(true, false);
      setStatus('AOI geändert. Bitte <b>Preview Overlay</b> erneut ausführen.', 'ok');
    });

    btnClear.addEventListener('click', clearAll);

    btnGJ.addEventListener('click', () => {
      if(!currentFeature) return;
      downloadText('aoi_epsg4326.geojson', elGeo.value);
    });

    async function preview(){
      if(!currentFeature) return;
      if (btnPreview.dataset.busy === "1") return;
      btnPreview.dataset.busy = "1";
      btnPreview.disabled = true;
      setButtons(true, false);
      setStatus('STAC Suche + Render läuft…', '');

      let gj;
      try{
        gj = JSON.parse(elGeo.value);
      }catch(err){
        setStatus('GeoJSON ist ungültig.', 'err');
        btnPreview.dataset.busy = "0";
        btnPreview.disabled = !currentFeature;
        return;
      }

      const n = Number(elN.value || 12);
      const orbit_state = String(elOrbit.value || "");
      const mode = String(elMode.value || "vv");

      try{
        const res = await fetch('/api/preview', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ geojson: gj, n, orbit_state, mode })
        });

        const ct = (res.headers.get('content-type') || '').toLowerCase();
        const raw = await res.text();

        let data = null;
        if (ct.includes('application/json')) {
          data = raw ? JSON.parse(raw) : {};
        } else {
          throw new Error(`Server lieferte kein JSON (HTTP ${res.status}, Content-Type=${ct}). Antwort-Auszug: ${raw.slice(0, 240)}`);
        }

        if (!res.ok) {
          throw new Error((data && data.error) ? data.error : (`HTTP ${res.status}`));
        }

        currentJob = data.job_id;

        if(overlay){
          map.removeLayer(overlay);
          overlay = null;
        }
        const b = data.overlay.bounds; // [[south, west],[north, east]]
        overlay = L.imageOverlay(data.overlay.url, b, { opacity: 1.0, interactive: false });
        overlay.addTo(map);

        map.fitBounds(L.latLngBounds(b).pad(0.15));

        setButtons(true, true);

        const t0 = data.t0 ? (`t0: ${data.t0.id} @ ${data.t0.datetime}`) : 't0: ?';
        const t1 = data.t1 ? (`t-1: ${data.t1.id} @ ${data.t1.datetime}`) : '';

        setStatus(
          `Overlay geladen (<b>${data.mode}</b>). AOI: ${data.aoi_area_km2.toFixed(3)} km²<br>${t0}${t1 ? '<br>'+t1 : ''}`,
          'ok'
        );

        elMeta.style.display = 'block';
        elMeta.textContent = JSON.stringify({ t0: data.t0, t1: data.t1, debug: data.debug }, null, 2);

        btnPNG.onclick = () => { window.location = data.download.png; };
        btnTIF.onclick = () => { window.location = data.download.geotiff; };

      }catch(err){
        setButtons(true, false);
        setStatus('Fehler: ' + (err && err.message ? err.message : String(err)), 'err');
      }finally {
        btnPreview.dataset.busy = "0";
        btnPreview.disabled = !currentFeature;
      }
    }

    btnPreview.addEventListener('click', preview);

    clearAll();
  </script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(
        INDEX_HTML,
        title=APP_TITLE,
        default_n=DEFAULT_N,
        max_n=MAX_N,
        max_area_km2=MAX_AOI_AREA_KM2,
        max_dim=MAX_RASTER_DIM_PX,
        db_factor=DB_FACTOR,
    )


@app.post("/api/search")
def api_search():
    """
    API: GeoJSON rein -> STAC Items (letzte N, optional orbit_state)
    Body: { "geojson": <Feature|FeatureCollection(1)|Polygon>, "n": 12, "orbit_state": ""|"ascending"|"descending" }
    """
    try:
        body = request.get_json(force=True, silent=False) or {}
        gj = _parse_geojson(body.get("geojson"))
        n = int(body.get("n", DEFAULT_N))
        orbit_state = str(body.get("orbit_state", "") or "")

        geom = _extract_single_geometry(gj)
        items, dbg = _stac_search(mapping(geom), n=n, orbit_state=orbit_state)

        # Return top N summaries after sorting
        out = [_item_summary(it) for it in items[:max(1, min(n, MAX_N))]]
        return jsonify({"items": out, "debug": dbg})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.post("/api/preview")
def api_preview():
    """
    API: GeoJSON rein -> PNG Overlay + GeoTIFF (VV/VH oder Δ(t0−t-1))
    Body: { "geojson": <Feature|FeatureCollection(1)|Polygon>, "n": 12, "orbit_state": "", "mode": "vv|vh|diff_vv|diff_vh" }
    """
    try:
        body = request.get_json(force=True, silent=False) or {}
        gj = _parse_geojson(body.get("geojson"))
        n = int(body.get("n", DEFAULT_N))
        orbit_state = str(body.get("orbit_state", "") or "")
        mode = str(body.get("mode", "vv") or "vv")

        rr = _render_from_geojson(gj, n=n, orbit_state=orbit_state, mode=mode)

        sw, ne = rr.bounds_wgs84
        return jsonify(
            {
                "job_id": rr.job_id,
                "mode": rr.mode,
                "aoi_area_km2": rr.aoi_area_km2,
                "overlay": {
                    "url": f"/r/{rr.job_id}/overlay.png",
                    "bounds": [[sw[0], sw[1]], [ne[0], ne[1]]],
                },
                "download": {
                    "png": f"/r/{rr.job_id}/overlay.png",
                    "geotiff": f"/r/{rr.job_id}/data.tif",
                },
                "t0": rr.t0,
                "t1": rr.t1,
                "debug": rr.debug,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/r/<job_id>/overlay.png")
def job_overlay(job_id: str):
    p = TMP_DIR / f"{job_id}.overlay.png"
    if not p.exists():
        return jsonify({"error": "Job nicht gefunden/abgelaufen."}), 404
    return send_file(p, mimetype="image/png", as_attachment=False, conditional=True)


@app.get("/r/<job_id>/data.tif")
def job_tif(job_id: str):
    p = TMP_DIR / f"{job_id}.data.tif"
    if not p.exists():
        return jsonify({"error": "Job nicht gefunden/abgelaufen."}), 404
    # epsg in filename if possible (best effort)
    return send_file(
        p,
        mimetype="image/tiff",
        as_attachment=True,
        download_name=f"s1_{job_id}_db_factor{int(DB_FACTOR)}.tif",
        conditional=True,
    )


@app.get("/healthz")
def healthz():
    return Response("ok", mimetype="text/plain")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
