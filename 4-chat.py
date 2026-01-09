import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import weaviate
from weaviate.auth import Auth

from pydantic import BaseModel, Field, ConfigDict

load_dotenv()

assert os.getenv("WEAVIATE_URL")
assert os.getenv("WEAVIATE_API_KEY")
assert os.getenv("OPENAI_API_KEY")

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
COLLECTION_NAME = "Guideline"


def retrieve_all_strategies(
    query: str,
    k: int = 5,
    alpha: float = 0.8
):
    rows = []

    with weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    ) as client:
        col = client.collections.use(COLLECTION_NAME)

        props = ["search_text", "breadcrumbs", "chunk_id", "source", "year"]

        resp = col.query.hybrid(query=query, alpha=alpha, limit=k, return_properties=props)
        rows.append(resp.objects)

    return rows