"""Health trends derived from MongoDB — no Snowflake required."""
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from dashboard.data.mongodb_reader import get_mongo_client
from dashboard.settings import get_settings

logger = logging.getLogger(__name__)


def fetch_health_trends(hours: int = 24) -> tuple[pd.DataFrame | None, bool]:
    try:
        settings = get_settings()
        client = get_mongo_client()
        collection = client[settings.MONGODB_DB][settings.MONGODB_COLLECTION]

        since = datetime.now(timezone.utc) - timedelta(hours=hours)

        pipeline = [
            {
                "$match": {
                    "type": "health",
                    "processed_at": {"$gte": since.isoformat()},
                }
            },
            {
                "$group": {
                    "_id": {
                        "hour": {
                            "$dateToString": {
                                "format": "%Y-%m-%d %H:00",
                                "date": {"$dateFromString": {"dateString": "$processed_at"}},
                            }
                        },
                        "subtype": "$subtype",
                    },
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"_id.hour": 1}},
        ]

        results = list(collection.aggregate(pipeline))

        if not results:
            return (pd.DataFrame(columns=["hour", "subtype", "count"]), False)

        df = pd.DataFrame([
            {"hour": r["_id"]["hour"], "subtype": r["_id"]["subtype"], "count": r["count"]}
            for r in results
        ])
        return (df, False)

    except Exception as e:
        logger.error("Health trends fetch failed: %s", e)
        return (None, True)
