import logging
import certifi
from pymongo import MongoClient
from dashboard.settings import get_settings

logger = logging.getLogger(__name__)
_cached_data: list[dict] = []


def get_mongo_client():
    settings = get_settings()
    client = MongoClient(settings.MONGODB_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
    return client


def fetch_latest_events(n: int = 50) -> tuple[list[dict], bool]:
    global _cached_data
    settings = get_settings()
    try:
        client = get_mongo_client()
        # Force a real connection attempt
        client.admin.command("ping")
        collection = client[settings.MONGODB_DB][settings.MONGODB_COLLECTION]
        docs = list(
            collection.find({}, {"_id": 0})
            .sort("processed_at", -1)
            .limit(n)
        )
        _cached_data = docs
        return (docs, False)
    except Exception as e:
        logger.error("MongoDB fetch failed: %s", e)
        return (_cached_data, True)
