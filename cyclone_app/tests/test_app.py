"""Smoke tests: python tests/test_app.py   (needs models/track/model.joblib)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from server import app  # noqa: E402

c = app.test_client()
assert c.get("/api/health").json["ok"]
cat = c.get("/api/cyclones").json
assert len(cat) > 300
fani = next(x for x in cat if x["name"] == "FANI")
cy = c.get(f"/api/cyclone/{fani['id']}").json
assert cy["lifecycle"]["landfall"]["time"].startswith("2019-05-03"), cy["lifecycle"]["landfall"]
f = c.get(f"/api/forecast/{fani['id']}?asof=2019-04-30T12:00").json
assert [p["h"] for p in f["forecast"]] == [6, 12, 18, 24, 30, 36, 42, 48]
assert f["landfall"] is not None
live = c.post("/api/forecast", json={"csv": "time_utc,lat,lon,wind_kt,ecp\n2024-05-25 12:00,18.8,89.5,35,989\n"
                                        "2024-05-25 18:00,19.3,89.4,40,988\n2024-05-26 00:00,20.0,89.2,50,983"}).json
assert len(live["forecast"]) == 8
assert c.post("/api/forecast", json={"csv": "a,b\n1,2"}).status_code == 400
assert c.get("/api/accuracy").json["cv"]
print("all tests passed")
