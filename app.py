#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import array_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds as win_from_bounds

from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer, Geod

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

APP_TITLE = os.getenv("APP_TITLE", "FieldLense – Sentinel-1 Radar (RTC, STAC)")

# STAC (Planetary Computer public endpoint)
STAC_API = os.getenv("STAC_API", "https://planetarycomputer.microsoft.com/api/stac/v1").rstrip("/")
STAC_COLLECTION = os.getenv("STAC_COLLECTION", "sentinel-1-rtc")  # DEFAULT: RTC

# SAS (Planetary Computer data auth)
PC_SAS_BASE = os.getenv("PC_SAS_BASE", "https://planetarycomputer.microsoft.com/api/sas/v1").rstrip("/")
PC_SAS_TOKEN_URL = os.getenv("PC_SAS_TOKEN_URL", f"{PC_SAS_BASE}/token/{{collection}}")
PC_SAS_SIGN_URL = os.getenv("PC_SAS_SIGN_URL", f"{PC_SAS_BASE}/sign")

# Account / Subscription key (needed for sentinel-1-rtc SAS tokens)
PC_SDK_SUBSCRIPTION_KEY = os.getenv("PC_SDK_SUBSCRIPTION_KEY", "").strip()

# Limits
MAX_AOI_AREA_KM2 = float(os.getenv("MAX_AOI_AREA_KM2", "25.0"))
DEFAULT_N = int(os.getenv("DEFAULT_N", "12"))
MAX_N = int(os.getenv("MAX_N", "50"))
MAX_RASTER_DIM_PX = int(os.getenv("MAX_RASTER_DIM_PX", "1536"))

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60"))

# dB conversion factor (power->dB: 10*log10, amplitude->dB: 20*log10)
DB_FACTOR = float(os.getenv("DB_FACTOR", "10.0"))

# Cache (/tmp on Cloud Run)
TMP_DIR = Path(os.getenv("TMP_DIR", "/tmp")) / "fl_s1_radar_cache"
TMP_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
MAX_CACHE_ITEMS = int(os.getenv("MAX_CACHE_ITEMS", "60"))

# Token cache
TOKEN_TTL_SECONDS = int(os.getenv("PC_TOKEN_TTL_SECONDS", "1800"))

# If you want to restrict by time window to reduce load (optional)
TIME_RANGE_DAYS = int(os.getenv("TIME_RANGE_DAYS", "180"))  # last 180d


# ------------------------------------------------------------
# Flask
# ------------------------------------------------------------

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
SESSION = requests.Session()

GEOD = Geod(ellps="WGS84")


# ------------------------------------------------------------
# Data structures
# ------------------------------------------------------------

@dataclass
class RenderResult:
    job_id: str
    bounds_wgs84: Tuple[Tuple[float, float], Tuple[float, float]]  # ((south, west), (north, east))
    mode: str
    n: int
    orbit_state: str
    overlay_png: Path
    out_tif: Path
    debug: Dict[str, Any]


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def _cleanup_cache() -> None:
    try:
        items = []
        for p in TMP_DIR.glob("*"):
            if p.is_file():
                items.append((p.stat().st_mtime, p))
        items.sort(reverse=True)

        now = time.time()
        for mtime, p in items:
            if now - mtime > CACHE_TTL_SECONDS:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

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
    raise ValueError(f"Nicht unterstützter GeoJSON-Typ: {t}.")


def _aoi_area_km2_geodesic(geom_wgs84) -> float:
    if geom_wgs84.is_empty:
        return 0.0
    try:
        area_m2, _ = GEOD.geometry_area_perimeter(geom_wgs84)
        return abs(float(area_m2)) / 1_000_000.0
    except Exception:
        # fallback: rough planar in EPSG:3857
        tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        g2 = shp_transform(lambda x, y: tr.transform(x, y), geom_wgs84)
        return abs(float(g2.area)) / 1_000_000.0


def _bounds_to_leaflet(bounds_wgs84: Tuple[float, float, float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    minx, miny, maxx, maxy = bounds_wgs84  # lon/lat
    # leaflet [[south, west],[north, east]]
    return (miny, minx), (maxy, maxx)


def _compute_scaled_dims(width_px: float, height_px: float, max_dim_px: int) -> Tuple[int, int]:
    if width_px <= 0 or height_px <= 0:
        raise ValueError("Ungültige Ausdehnung (Breite/Höhe <= 0).")
    if width_px >= height_px:
        w = max_dim_px
        h = max(1, int(round(max_dim_px * (height_px / width_px))))
    else:
        h = max_dim_px
        w = max(1, int(round(max_dim_px * (width_px / height_px))))
    return w, h


# ------------------------------------------------------------
# Planetary Computer SAS handling
# ------------------------------------------------------------

_TOKEN_CACHE: Dict[str, Tuple[float, str]] = {}  # collection -> (expires_epoch, token_string)


def _pc_headers() -> Dict[str, str]:
    # Subscription key is required for certain datasets (RTC). :contentReference[oaicite:1]{index=1}
    if PC_SDK_SUBSCRIPTION_KEY:
        return {"Ocp-Apim-Subscription-Key": PC_SDK_SUBSCRIPTION_KEY}
    return {}


def _pc_get_collection_token(collection: str) -> str:
    now = time.time()
    if collection in _TOKEN_CACHE:
        exp, tok = _TOKEN_CACHE[collection]
        if now < exp - 30:
            return tok

    url = PC_SAS_TOKEN_URL.format(collection=collection)
    r = SESSION.get(url, headers=_pc_headers(), timeout=HTTP_TIMEOUT)
    if r.status_code in (401, 403):
        # For sentinel-1-rtc this is expected without a key. :contentReference[oaicite:2]{index=2}
        raise ValueError(
            f"SAS-Token Zugriff verweigert (HTTP {r.status_code}). "
            f"Für '{collection}' benötigst du i.d.R. PC_SDK_SUBSCRIPTION_KEY (Planetary Computer Developer Portal Key)."
        )
    if not r.ok:
        raise ValueError(f"SAS-Token Request fehlgeschlagen (HTTP {r.status_code}): {r.text[:500]}")

    data = r.json()
    tok = data.get("token")
    if not isinstance(tok, str) or not tok.strip():
        raise ValueError("SAS-Token Antwort hat kein gültiges 'token' Feld (String).")

    tok = tok.strip()
    _TOKEN_CACHE[collection] = (now + TOKEN_TTL_SECONDS, tok)
    return tok


def _pc_sign_href(href: str) -> str:
    """
    Returns a signed href.
    Strategy:
      1) Try /sign endpoint (returns {"href": "..."}).
      2) Fallback: append collection-level token.
    """
    if not href:
        return href
    hlow = href.lower()
    if "sig=" in hlow and "sv=" in hlow:
        return href

    # Try sign endpoint first
    try:
        r = SESSION.get(PC_SAS_SIGN_URL, params={"href": href}, headers=_pc_headers(), timeout=HTTP_TIMEOUT)
        if r.ok:
            j = r.json()
            if isinstance(j, dict) and isinstance(j.get("href"), str) and j["href"].strip():
                return j["href"].strip()
    except Exception:
        pass

    # Fallback: token append
    tok = _pc_get_collection_token(STAC_COLLECTION)
    sep = "&" if "?" in href else "?"
    # token strings from PC are typically query fragments without leading '?'
    return href + sep + tok


# ------------------------------------------------------------
# STAC
# ------------------------------------------------------------

def _iso_now_minus_days(days: int) -> str:
    # STAC datetime interval: start/end
    end = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start_epoch = time.time() - float(days) * 86400.0
    start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_epoch))
    return f"{start}/{end}"


def _stac_search(
    geom_wgs84,
    n: int,
    orbit_state: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    n = max(1, min(int(n), MAX_N))
    url = f"{STAC_API}/search"

    body: Dict[str, Any] = {
        "collections": [STAC_COLLECTION],
        "intersects": mapping(geom_wgs84),
        "limit": max(n * 4, 20),
        "datetime": _iso_now_minus_days(TIME_RANGE_DAYS),
    }

    # Some catalogs support query on sat:orbit_state / s1:orbit_state etc.
    # We keep it best-effort and apply a strict client-side filter afterwards.
    if orbit_state:
        body["query"] = {
            "sat:orbit_state": {"eq": orbit_state.lower()},
        }

    r = SESSION.post(url, json=body, timeout=HTTP_TIMEOUT)
    if not r.ok:
        try:
            j = r.json()
        except Exception:
            j = r.text[:800]
        raise ValueError(f"STAC Request fehlgeschlagen (HTTP {r.status_code}): {j}")

    resp = r.json()
    feats = resp.get("features") or []

    # Client-side filter/sort
    items = []
    for it in feats:
        p = it.get("properties") or {}
        dt = p.get("datetime") or p.get("start_datetime") or ""
        if not dt:
            continue
        if orbit_state:
            # accept several possible locations
            os1 = (p.get("sat:orbit_state") or p.get("s1:orbit_state") or p.get("orbit_state") or "").lower()
            if os1 != orbit_state.lower():
                continue
        items.append(it)

    def _dt_key(it):
        p = it.get("properties") or {}
        return p.get("datetime") or ""

    items.sort(key=_dt_key, reverse=True)
    items = items[: max(n, 2)]  # for diff we need at least 2

    debug = {
        "stac_url": url,
        "stac_body": body,
        "returned": len(feats),
        "kept_after_filter": len(items),
    }
    return items, debug


# ------------------------------------------------------------
# Asset picking + STAC proj:* fallback
# ------------------------------------------------------------

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


def _asset_candidates(item: Dict[str, Any], pol: str) -> List[Dict[str, Any]]:
    assets = item.get("assets") or {}
    pol = (pol or "").lower().strip()
    cands = []
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

        cands.append({
            "key": str(k),
            "href": str(href),
            "type": str(a.get("type") or ""),
            "roles": roles,
            "title": str(a.get("title") or ""),
            "score": score,
            "asset": a,
        })
    cands.sort(key=lambda d: d["score"], reverse=True)
    return cands


def _stac_proj_info(item: dict, asset: dict) -> Tuple[Optional[CRS], Optional[rasterio.Affine], Optional[Tuple[int, int]]]:
    props = item.get("properties", {}) or {}

    code = asset.get("proj:code") or props.get("proj:code")
    epsg = asset.get("proj:epsg") or props.get("proj:epsg")
    wkt2 = asset.get("proj:wkt2") or props.get("proj:wkt2")
    tr = asset.get("proj:transform") or props.get("proj:transform")
    sh = asset.get("proj:shape") or props.get("proj:shape")

    crs = None
    if isinstance(code, str) and code.strip():
        crs = CRS.from_string(code.strip())
    elif epsg is not None:
        crs = CRS.from_epsg(int(epsg))
    elif isinstance(wkt2, str) and wkt2.strip():
        crs = CRS.from_wkt(wkt2)

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
        shape_yx = (int(sh[0]), int(sh[1]))

    return crs, aff, shape_yx


# ------------------------------------------------------------
# Core raster read: pol -> dB on own grid or dst grid
# ------------------------------------------------------------

def _read_pol_db_on_grid(
    item: Dict[str, Any],
    pol: str,
    aoi_geom_wgs84,
    dst_crs=None,
    dst_transform=None,
    dst_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, Any, Any, Tuple[float, float, float, float], Dict[str, Any]]:
    """
    Returns:
      arr_db(float32 with NaN), crs, transform, bounds(left,bottom,right,top) in crs, debug_used
    """

    cands = _asset_candidates(item, pol)
    if not cands:
        raise ValueError(f"Kein geeignetes GeoTIFF-Asset für '{pol}' gefunden. Assets: {list((item.get('assets') or {}).keys())[:40]}")

    last_err = None
    b_wgs = aoi_geom_wgs84.bounds  # lon/lat

    for cand in cands:
        asset_key = cand["key"]
        asset = cand["asset"]
        href = cand["href"]

        signed = _pc_sign_href(href)

        try:
            with rasterio.Env(
                GDAL_DISABLE_READDIR_ON_OPEN="YES",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
            ):
                with rasterio.open(signed) as src:
                    crs = src.crs
                    transform = src.transform

                    # If CRS/transform are missing in file header, try STAC proj:* metadata
                    if crs is None or transform is None or transform == rasterio.Affine.identity():
                        stac_crs, stac_transform, _stac_shape = _stac_proj_info(item, asset)
                        if stac_crs is None or stac_transform is None:
                            raise ValueError(f"Asset '{asset_key}' hat keine Geo-Referenz im File und keine proj:* Metadaten.")
                        crs = stac_crs
                        transform = stac_transform

                    # AOI bounds into raster CRS
                    bb = transform_bounds("EPSG:4326", crs, b_wgs[0], b_wgs[1], b_wgs[2], b_wgs[3], densify_pts=21)
                    left, bottom, right, top = bb
                    if right < left:
                        left, right = right, left
                    if top < bottom:
                        bottom, top = top, bottom

                    # Window from bounds; fallback via inverse affine if needed
                    try:
                        w = win_from_bounds(left, bottom, right, top, transform=transform)
                    except Exception:
                        inv = ~transform
                        c0, r0 = inv * (left, top)
                        c1, r1 = inv * (right, bottom)
                        rmin, rmax = sorted([int(math.floor(r0)), int(math.ceil(r1))])
                        cmin, cmax = sorted([int(math.floor(c0)), int(math.ceil(c1))])
                        w = Window(col_off=cmin, row_off=rmin, width=max(0, cmax - cmin), height=max(0, rmax - rmin))

                    w = w.round_offsets().round_lengths()
                    w = w.intersection(Window(0, 0, src.width, src.height))
                    if w.width <= 0 or w.height <= 0:
                        raise ValueError("AOI liegt außerhalb der Szene (Window leer).")

                    # output shape
                    if dst_shape is None:
                        out_w, out_h = _compute_scaled_dims(float(w.width), float(w.height), MAX_RASTER_DIM_PX)
                        oh, ow = out_h, out_w
                    else:
                        oh, ow = int(dst_shape[0]), int(dst_shape[1])

                    win_transform = rasterio.windows.transform(w, transform)
                    scale_x = float(w.width) / float(ow)
                    scale_y = float(w.height) / float(oh)
                    out_transform = win_transform * rasterio.Affine.scale(scale_x, scale_y)

                    arr = src.read(
                        1,
                        window=w,
                        out_shape=(oh, ow),
                        resampling=Resampling.bilinear,
                    ).astype(np.float32)

                    # linear -> dB
                    arr[arr <= 0] = np.nan
                    arr_db = DB_FACTOR * np.log10(arr)

                    # mask outside AOI
                    try:
                        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
                        aoi_proj = shp_transform(lambda x, y: tr.transform(x, y), aoi_geom_wgs84)
                        m = geometry_mask(
                            [mapping(aoi_proj)],
                            out_shape=(oh, ow),
                            transform=out_transform,
                            invert=True,
                            all_touched=False,
                        )
                        arr_db[~m] = np.nan
                    except Exception:
                        pass

                    # reproject to dst grid if requested
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

                    # bounds for output grid
                    h, w2 = arr_db.shape
                    b2 = array_bounds(h, w2, out_transform)  # (south, west, north, east)?? actually returns (ymin,xmin,ymax,xmax)?? rasterio returns (bottom, left, top, right)
                    # rasterio.transform.array_bounds returns (ymin, xmin, ymax, xmax) if transform is standard; but for our usage we want (left,bottom,right,top).
                    # safer: compute using dataset-style: (left, bottom, right, top)
                    ymin, xmin, ymax, xmax = b2
                    bounds = (xmin, ymin, xmax, ymax)

                    debug_used = {
                        "asset_key": asset_key,
                        "asset_type": cand.get("type", ""),
                        "asset_roles": cand.get("roles", []),
                        "href": href,
                        "signed_href": signed[:180] + ("…" if len(signed) > 180 else ""),
                        "crs": str(crs),
                        "transform": [out_transform.a, out_transform.b, out_transform.c, out_transform.d, out_transform.e, out_transform.f],
                        "window": {"col_off": float(w.col_off), "row_off": float(w.row_off), "width": float(w.width), "height": float(w.height)},
                    }

                    return arr_db, crs, out_transform, bounds, debug_used

        except Exception as e:
            last_err = e
            continue

    raise ValueError(f"Kein brauchbares '{pol}' Asset. Letzter Fehler: {last_err}. Kandidaten: {[c['key'] for c in cands[:10]]}")


# ------------------------------------------------------------
# Rendering helpers
# ------------------------------------------------------------

def _nan_percentiles(a: np.ndarray, ps: List[float]) -> List[float]:
    v = a[np.isfinite(a)]
    if v.size < 10:
        return [float("nan") for _ in ps]
    return [float(np.nanpercentile(v, p)) for p in ps]


def _to_png_rgba(data: np.ndarray, mode: str) -> Image.Image:
    """
    Create an RGBA overlay image.
    - vv/vh: grayscale stretched by p2..p98
    - diff: 0 centered, symmetric stretch by max(|p2|,|p98|)
    """
    a = data.astype(np.float32)
    alpha = np.isfinite(a).astype(np.uint8) * 255

    if mode.startswith("diff"):
        p2, p98 = _nan_percentiles(a, [2, 98])
        if not np.isfinite(p2) or not np.isfinite(p98):
            lo, hi = -1.0, 1.0
        else:
            m = max(abs(p2), abs(p98))
            lo, hi = -m, +m
        x = np.clip((a - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        # grayscale with midpoint ~0 => 0.5
        g = (x * 255.0).astype(np.uint8)
        rgb = np.dstack([g, g, g])
    else:
        p2, p98 = _nan_percentiles(a, [2, 98])
        if not np.isfinite(p2) or not np.isfinite(p98) or p98 <= p2:
            p2, p98 = -25.0, 5.0
        x = np.clip((a - p2) / (p98 - p2 + 1e-9), 0.0, 1.0)
        g = (x * 255.0).astype(np.uint8)
        rgb = np.dstack([g, g, g])

    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def _write_float_geotiff(path: Path, data: np.ndarray, crs: CRS, transform: rasterio.Affine) -> None:
    nodata = -9999.0
    out = data.copy()
    out[~np.isfinite(out)] = nodata
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
        dst.write(out.astype(np.float32), 1)


# ------------------------------------------------------------
# High-level render pipeline
# ------------------------------------------------------------

def _select_t0_t1(items: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    if not items:
        raise ValueError("Keine Items gefunden.")
    t0 = items[0]
    # prefer same platform + relative orbit if possible
    p0 = t0.get("properties") or {}
    rel0 = p0.get("sat:relative_orbit") or p0.get("s1:relative_orbit") or p0.get("relative_orbit")
    plat0 = p0.get("platform") or ""
    orb0 = p0.get("sat:orbit_state") or p0.get("s1:orbit_state") or p0.get("orbit_state") or ""

    for it in items[1:]:
        p = it.get("properties") or {}
        if plat0 and (p.get("platform") or "") != plat0:
            continue
        if rel0 is not None and (p.get("sat:relative_orbit") or p.get("s1:relative_orbit") or p.get("relative_orbit")) != rel0:
            continue
        if orb0 and (p.get("sat:orbit_state") or p.get("s1:orbit_state") or p.get("orbit_state") or "") != orb0:
            continue
        return t0, it

    return t0, (items[1] if len(items) > 1 else None)


def _render_from_geojson(geojson: Dict[str, Any], mode: str, n: int, orbit_state: str) -> RenderResult:
    _cleanup_cache()

    if STAC_COLLECTION == "sentinel-1-rtc" and not PC_SDK_SUBSCRIPTION_KEY:
        # sentinel-1-rtc requires an account/subscription key for SAS tokens. :contentReference[oaicite:3]{index=3}
        raise ValueError("sentinel-1-rtc benötigt PC_SDK_SUBSCRIPTION_KEY (Planetary Computer Developer Portal Key) für SAS-Tokens.")

    geom_wgs84 = _extract_single_geometry(geojson)
    if geom_wgs84.is_empty:
        raise ValueError("Geometrie ist leer.")
    if geom_wgs84.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError("Nur Polygon/MultiPolygon erlaubt.")

    aoi_area_km2 = _aoi_area_km2_geodesic(geom_wgs84)
    if MAX_AOI_AREA_KM2 > 0 and aoi_area_km2 > MAX_AOI_AREA_KM2:
        raise ValueError(f"AOI ist zu groß: {aoi_area_km2:.3f} km² (Limit: {MAX_AOI_AREA_KM2:.3f} km²).")

    n = max(1, min(int(n), MAX_N))
    mode = (mode or "vv").lower().strip()
    orbit_state = (orbit_state or "").lower().strip()

    # STAC items
    items, stac_debug = _stac_search(geom_wgs84, n=n, orbit_state=orbit_state)
    if not items:
        raise ValueError("Keine Szenen gefunden (Filter zu strikt oder Zeitfenster zu klein).")

    # Choose t0 / t-1
    t0, t1 = _select_t0_t1(items)

    debug: Dict[str, Any] = {
        "collection": STAC_COLLECTION,
        "mode": mode,
        "n": n,
        "orbit_state": orbit_state,
        "aoi_area_km2": aoi_area_km2,
        "stac": stac_debug,
        "t0": {
            "id": t0.get("id"),
            "datetime": (t0.get("properties") or {}).get("datetime"),
        },
        "t1": None,
    }
    if t1:
        debug["t1"] = {
            "id": t1.get("id"),
            "datetime": (t1.get("properties") or {}).get("datetime"),
        }

    # Read data
    pol = "vv" if "vv" in mode else "vh"
    if mode in ("vv", "vh"):
        arr0, crs0, tr0, bounds0, used0 = _read_pol_db_on_grid(t0, pol, geom_wgs84)
        out = arr0
        out_crs, out_tr, out_bounds = crs0, tr0, bounds0
        debug["used_assets"] = {"t0": used0}
    elif mode in ("diff_vv", "diff_vh"):
        if not t1:
            raise ValueError("Für Differenz wird mindestens eine zweite Szene benötigt (N erhöhen / Filter lockern).")
        arr0, crs0, tr0, bounds0, used0 = _read_pol_db_on_grid(t0, pol, geom_wgs84)
        arr1, _, _, _, used1 = _read_pol_db_on_grid(t1, pol, geom_wgs84, dst_crs=crs0, dst_transform=tr0, dst_shape=arr0.shape)
        out = arr0 - arr1
        out_crs, out_tr, out_bounds = crs0, tr0, bounds0
        debug["used_assets"] = {"t0": used0, "t1": used1}
    else:
        raise ValueError("mode ungültig. Erlaubt: vv, vh, diff_vv, diff_vh")

    # Create outputs
    job_id = uuid.uuid4().hex[:12]
    out_png = TMP_DIR / f"{job_id}.overlay.png"
    out_tif = TMP_DIR / f"{job_id}.data.tif"

    img = _to_png_rgba(out, mode=mode)
    img.save(out_png)

    _write_float_geotiff(out_tif, out, out_crs, out_tr)

    # Bounds -> WGS84 leaflet overlay bounds
    # out_bounds are (minx,miny,maxx,maxy) in out_crs, convert to EPSG:4326
    minx, miny, maxx, maxy = out_bounds
    wgs_bounds = transform_bounds(out_crs, CRS.from_epsg(4326), minx, miny, maxx, maxy, densify_pts=21)
    sw, ne = _bounds_to_leaflet(wgs_bounds)

    return RenderResult(
        job_id=job_id,
        bounds_wgs84=(sw, ne),
        mode=mode,
        n=n,
        orbit_state=orbit_state,
        overlay_png=out_png,
        out_tif=out_tif,
        debug=debug,
    )


# ------------------------------------------------------------
# HTML UI
# ------------------------------------------------------------

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
    header{
      max-width: var(--container);
      margin: 18px auto 0;
      padding: 0 14px;
      display:flex; align-items:baseline; justify-content:space-between; gap: 12px;
    }
    h1{ font-size: 18px; margin:0; letter-spacing: .2px; }
    .hint{ color: var(--muted); font-size: 13px; margin-top: 6px; line-height: 1.35; }
    .wrap{
      max-width: var(--container);
      margin: 18px auto;
      padding: 0 14px 24px;
      display: grid;
      grid-template-columns: 1.2fr .8fr;
      gap: var(--gap);
    }
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
      min-height: 240px;
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
    button{
      appearance:none;
      border: 1px solid var(--border);
      background: rgba(255,255,255,.06);
      color: var(--text);
      padding: 10px 12px;
      border-radius: 12px;
      cursor: pointer;
      font-weight: 600;
    }
    button.primary{ border-color: rgba(110,168,254,.35); background: rgba(110,168,254,.16); }
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
    select, input[type="number"]{
      padding: 8px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,.04);
      color: var(--text);
      outline: none;
    }
    .chapter{
      padding: 10px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: rgba(0,0,0,.14);
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    .chapter b{ color: var(--text); }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>{{ title }}</h1>
      <div class="hint">
        Zeichne eine AOI (Polygon/Rechteck). Es ist immer nur <b>ein</b> Feature aktiv – neues Feature ersetzt das alte.
        <br/>Dataset: <b>{{ collection }}</b> · API: <code>/api/preview</code>, <code>/api/search</code>
      </div>
    </div>
    <div class="small">Health: <code>/healthz</code></div>
  </header>

  <div class="wrap">
    <div class="card">
      <div id="map"></div>
    </div>

    <div class="card">
      <div class="panel">
        <div class="row">
          <button id="btn-clear">AOI löschen</button>
          <button class="primary" id="btn-render" disabled>Preview Overlay</button>
        </div>

        <div class="row">
          <label>Mode:
            <select id="mode" style="margin-left:8px;">
              <option value="vv">VV (t0)</option>
              <option value="vh">VH (t0)</option>
              <option value="diff_vv">ΔVV (t0 − t-1)</option>
              <option value="diff_vh">ΔVH (t0 − t-1)</option>
            </select>
          </label>

          <label>N:
            <input id="n" type="number" min="1" max="{{ max_n }}" value="{{ default_n }}" style="width:90px; margin-left:8px;">
          </label>

          <label>Orbit:
            <select id="orbit" style="margin-left:8px;">
              <option value="">(alle)</option>
              <option value="ascending">ascending</option>
              <option value="descending">descending</option>
            </select>
          </label>
        </div>

        <div id="status" class="status">Noch keine AOI.</div>

        <div class="row">
          <button id="btn-geojson" disabled>GeoJSON</button>
          <button id="btn-png" disabled>PNG</button>
          <button id="btn-tif" disabled>GeoTIFF</button>
        </div>

        <div class="chapter">
          <b>All-Weather (Radar)</b><br/>
          Sentinel-1 ist C-Band SAR: unabhängig von Tageslicht und weitgehend unempfindlich gegenüber Wolken.
          Deshalb eignet sich RTC besonders gut für regelmäßige Beobachtung, wenn optische Daten ausfallen.
        </div>

        <label>GeoJSON (aktuelles Feature, EPSG:4326)</label>
        <textarea id="geojson" spellcheck="false" placeholder="Hier erscheint das GeoJSON…"></textarea>

        <label>Debug (Request / Asset)</label>
        <textarea id="debug" spellcheck="false" placeholder="Hier erscheinen Debug-Infos (STAC/SAS)…"></textarea>

        <div class="small">
          Limit serverseitig: <b>{{ max_area_km2 }} km²</b> · Max Preview-Auflösung: <b>{{ max_dim }} px</b>
        </div>
      </div>
    </div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>

  <script>
    const map = L.map('map', { preferCanvas: true }).setView([49.87, 8.65], 11);

    const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 20,
      attribution: '&copy; OpenStreetMap'
    }).addTo(map);

    const drawn = new L.FeatureGroup().addTo(map);

    const drawControl = new L.Control.Draw({
      draw: {
        polyline: false, circle: false, circlemarker: false, marker: false,
        polygon: { allowIntersection: false, showArea: true },
        rectangle: true
      },
      edit: { featureGroup: drawn, edit: true, remove: false }
    });
    map.addControl(drawControl);

    let overlay = null;
    let currentFeature = null;
    let currentJob = null;

    const elGeo = document.getElementById('geojson');
    const elDbg = document.getElementById('debug');
    const elStatus = document.getElementById('status');
    const btnRender = document.getElementById('btn-render');
    const btnClear = document.getElementById('btn-clear');
    const btnGJ = document.getElementById('btn-geojson');
    const btnPNG = document.getElementById('btn-png');
    const btnTIF = document.getElementById('btn-tif');

    const elMode = document.getElementById('mode');
    const elN = document.getElementById('n');
    const elOrbit = document.getElementById('orbit');

    function setStatus(html, cls){
      elStatus.className = 'status' + (cls ? (' ' + cls) : '');
      elStatus.innerHTML = html;
    }
    function setButtons(hasFeature, hasJob){
      btnRender.disabled = !hasFeature;
      btnGJ.disabled = !hasFeature;
      btnPNG.disabled = !hasJob;
      btnTIF.disabled = !hasJob;
    }
    function clearAll(){
      drawn.clearLayers();
      currentFeature = null;
      currentJob = null;
      elGeo.value = '';
      elDbg.value = '';
      if(overlay){ map.removeLayer(overlay); overlay = null; }
      setButtons(false, false);
      setStatus('Noch keine AOI.', '');
    }
    function featureToGeoJSON(layer){
      return { type: "Feature", properties: { epsg: 4326 }, geometry: layer.toGeoJSON().geometry };
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
      if(overlay){ map.removeLayer(overlay); overlay = null; }
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
      if(overlay){ map.removeLayer(overlay); overlay = null; }
      setButtons(true, false);
      setStatus('AOI geändert. Bitte <b>Preview Overlay</b> erneut ausführen.', 'ok');
    });

    btnClear.addEventListener('click', clearAll);
    btnGJ.addEventListener('click', () => {
      if(!currentFeature) return;
      downloadText('aoi_epsg4326.geojson', elGeo.value);
    });

    async function renderPreview(){
      if(!currentFeature) return;
      if (btnRender.dataset.busy === "1") return;
      btnRender.dataset.busy = "1";
      btnRender.disabled = true;
      setButtons(true, false);

      const mode = elMode.value;
      const n = Number(elN.value || 12);
      const orbit_state = elOrbit.value;

      let gj;
      try{
        gj = JSON.parse(elGeo.value);
      }catch(err){
        setStatus('GeoJSON ist ungültig.', 'err');
        btnRender.dataset.busy = "0";
        btnRender.disabled = !currentFeature;
        return;
      }

      setStatus(`Request: <b>${mode}</b> · N=${n} · orbit=${orbit_state || "(alle)"} …`, '');

      try{
        const res = await fetch('/api/preview', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ geojson: gj, mode, n, orbit_state })
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

        // Debug box
        elDbg.value = JSON.stringify(data.debug || {}, null, 2);

        if(overlay){ map.removeLayer(overlay); overlay = null; }
        const b = data.overlay.bounds; // [[south, west],[north, east]]
        overlay = L.imageOverlay(data.overlay.url, b, { opacity: 1.0, interactive: false });
        overlay.addTo(map);

        const fit = L.latLngBounds(b);
        map.fitBounds(fit.pad(0.15));

        setButtons(true, true);
        setStatus(
          `Overlay geladen. <b>AOI</b>: ${data.aoi_area_km2.toFixed(3)} km² · <b>${data.mode}</b>`,
          'ok'
        );

        btnPNG.onclick = () => { window.location = data.download.png; };
        btnTIF.onclick = () => { window.location = data.download.geotiff; };

      }catch(err){
        setButtons(true, false);
        setStatus('Fehler: ' + (err && err.message ? err.message : String(err)), 'err');
      }finally {
        btnRender.dataset.busy = "0";
        btnRender.disabled = !currentFeature;
      }
    }

    btnRender.addEventListener('click', renderPreview);

    clearAll();
  </script>
</body>
</html>
"""


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.get("/")
def index():
    return render_template_string(
        INDEX_HTML,
        title=APP_TITLE,
        collection=STAC_COLLECTION,
        default_n=DEFAULT_N,
        max_n=MAX_N,
        max_area_km2=MAX_AOI_AREA_KM2,
        max_dim=MAX_RASTER_DIM_PX,
    )


@app.post("/api/search")
def api_search():
    try:
        body = request.get_json(force=True, silent=False) or {}
        gj = _parse_geojson(body.get("geojson"))
        geom = _extract_single_geometry(gj)

        n = int(body.get("n", DEFAULT_N))
        orbit_state = (body.get("orbit_state") or "").strip()

        items, dbg = _stac_search(geom, n=n, orbit_state=orbit_state)

        out = []
        for it in items:
            p = it.get("properties") or {}
            out.append({
                "id": it.get("id"),
                "datetime": p.get("datetime"),
                "platform": p.get("platform"),
                "orbit_state": p.get("sat:orbit_state") or p.get("s1:orbit_state") or p.get("orbit_state"),
                "relative_orbit": p.get("sat:relative_orbit") or p.get("s1:relative_orbit") or p.get("relative_orbit"),
                "assets": list((it.get("assets") or {}).keys())[:50],
            })

        return jsonify({"items": out, "debug": dbg})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.post("/api/preview")
def api_preview():
    try:
        body = request.get_json(force=True, silent=False) or {}
        gj = _parse_geojson(body.get("geojson"))
        mode = body.get("mode", "vv")
        n = int(body.get("n", DEFAULT_N))
        orbit_state = body.get("orbit_state", "")

        rr = _render_from_geojson(gj, mode=mode, n=n, orbit_state=orbit_state)

        sw, ne = rr.bounds_wgs84
        return jsonify({
            "job_id": rr.job_id,
            "mode": rr.mode,
            "n": rr.n,
            "orbit_state": rr.orbit_state,
            "aoi_area_km2": rr.debug.get("aoi_area_km2", 0.0),
            "overlay": {
                "url": f"/r/{rr.job_id}/overlay.png",
                "bounds": [[sw[0], sw[1]], [ne[0], ne[1]]],
            },
            "download": {
                "png": f"/r/{rr.job_id}/overlay.png",
                "geotiff": f"/r/{rr.job_id}/data.tif",
            },
            "debug": rr.debug,
        })

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
    return send_file(
        p,
        mimetype="image/tiff",
        as_attachment=True,
        download_name=f"s1_{job_id}_{STAC_COLLECTION}.tif",
        conditional=True,
    )


@app.get("/healthz")
def healthz():
    return Response("ok", mimetype="text/plain")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
