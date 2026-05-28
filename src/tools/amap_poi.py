import httpx
from typing import Optional
from src.config import settings


class AmapPOISearchTool:
    POI_TYPES = {
        "pharmacy": "010901",
        "hospital": "090100",
        "emergency": "090101",
        "clinic": "090200",
    }

    STRATEGY = {
        "mild": {"types": "010901", "radius": 1000},
        "moderate": {"types": "090100|090200", "radius": 3000},
        "severe": {"types": "090101|090100", "radius": 5000},
        "unknown": {"types": "090100|090200", "radius": 3000},
    }

    ENDPOINT = "https://restapi.amap.com/v3/place/around"

    async def search(
        self,
        location: tuple[float, float],
        severity: str = "unknown",
        keywords: Optional[str] = None,
    ) -> list[dict]:
        strategy = self.STRATEGY.get(severity, self.STRATEGY["unknown"])
        params = {
            "key": settings.amap_api_key,
            "location": f"{location[0]},{location[1]}",
            "radius": strategy["radius"],
            "types": strategy["types"],
            "offset": 5,
            "extensions": "all",
            "sortrule": "distance",
        }
        if keywords:
            params["keywords"] = keywords
            params.pop("types", None)

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(self.ENDPOINT, params=params)
            data = resp.json()

        if data.get("status") != "1":
            return []

        pois = data.get("pois", [])
        return [
            {
                "name": p.get("name", ""),
                "address": p.get("address", ""),
                "location": p.get("location", ""),
                "distance": p.get("distance", ""),
                "type": p.get("type", ""),
                "tel": p.get("tel", ""),
                "rating": p.get("biz_ext", {}).get("rating", ""),
            }
            for p in pois[:5]
        ]


amap_tool = AmapPOISearchTool()