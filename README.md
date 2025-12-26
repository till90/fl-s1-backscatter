# fl-s1-radar (STAC) – Sentinel-1 Radar (VV/VH + Δ t0−t-1)

Microservice für **Sentinel-1 GRD** über **STAC**:
- Browser-UI: AOI zeichnen (Polygon/Rechteck), Preview Overlay (PNG)
- API: Preview + Suche
- Modi:
  - `vv` / `vh` (t0)
  - `diff_vv` / `diff_vh` = Δ(t0 − t-1) in dB

## Deploy (Google Cloud Run, Source Deploy)
Dieses Repository enthält ein **einzelnes Python-Skript** (z. B. `main.py`) und eine `requirements.txt`.
Cloud Run erkennt Flask-Apps i. d. R. automatisch; sonst kannst du per Start Command z. B. `python main.py` verwenden.

Healthcheck:
- `GET /healthz` → `ok`

---

## ENV Variablen (optional)
| Variable | Default | Beschreibung |
|---|---:|---|
| `APP_TITLE` | FieldLense – Sentinel-1 Radar (STAC) | Titel in der UI |
| `STAC_API` | `https://planetarycomputer.microsoft.com/api/stac/v1` | STAC Endpoint |
| `STAC_COLLECTION` | `sentinel-1-grd` | Collection ID |
| `PC_SAS_TOKEN_URL` | `https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}` | SAS Token Endpoint (Planetary Computer) |
| `PC_TOKEN_TTL_SECONDS` | `1800` | Token Cache TTL |
| `DEFAULT_N` | `12` | Default „letzte N Szenen“ |
| `MAX_N` | `50` | Obergrenze N |
| `MAX_AOI_AREA_KM2` | `25.0` | AOI-Limit |
| `MAX_RASTER_DIM_PX` | `1536` | Max Preview/Output Dimension (Seitenlänge) |
| `HTTP_TIMEOUT` | `60` | Requests Timeout |
| `DB_FACTOR` | `10.0` | dB-Faktor: `10` (Power->dB) oder `20` (Amplitude->dB) |
| `TMP_DIR` | `/tmp` | Cache-Verzeichnis |
| `CACHE_TTL_SECONDS` | `3600` | Cache TTL |
| `MAX_CACHE_ITEMS` | `60` | Cache Size Cap |

---

## Browser UI
Öffne:
- `GET /`

Workflow:
1. AOI zeichnen
2. `Preview Overlay` klicken
3. Download Buttons nutzen (PNG/GeoTIFF)

---

## API

### 1) STAC-Suche (letzte N)
**POST** `/api/search`

Request:
```json
{
  "geojson": {
    "type": "Feature",
    "properties": { "epsg": 4326 },
    "geometry": {
      "type": "Polygon",
      "coordinates": [[[8.6,49.9],[8.9,49.9],[8.9,50.1],[8.6,50.1],[8.6,49.9]]]
    }
  },
  "n": 12,
  "orbit_state": "ascending"
}
