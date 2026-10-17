import json, urllib.request, urllib.parse

URL = ("https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
       "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V5/FeatureServer/0/query")
GM = ["Bolton","Bury","Manchester","Oldham","Rochdale","Salford",
      "Stockport","Tameside","Trafford","Wigan"]
where = " OR ".join(f"LSOA21NM LIKE '{la} %'" for la in GM)
OUT = "/home/ibm-user/shared/zcold/data/processed/lsoa_gm.geojson"

features, offset = [], 0
while True:
    q = urllib.parse.urlencode({"where": where, "outFields": "LSOA21CD,LSOA21NM",
        "outSR": 4326, "f": "geojson", "resultOffset": offset, "resultRecordCount": 1000})
    data = json.load(urllib.request.urlopen(f"{URL}?{q}", timeout=120))
    batch = data.get("features", [])
    features += batch
    print(f"fetched {len(batch)} (total {len(features)})")
    if len(batch) < 1000: break
    offset += 1000

json.dump({"type": "FeatureCollection", "features": features}, open(OUT, "w"))
print(f"Saved {len(features)} LSOAs -> {OUT}")
