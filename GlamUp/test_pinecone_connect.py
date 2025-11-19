from pinecone import Pinecone
import os

api_key = os.environ.get("PINECONE_API_KEY")
index_name = os.environ.get("PINECONE_INDEX_NAME", "slaylist-products")

print("API key prefix:", api_key[:7] if api_key else None)
print("Index name:", index_name)

pc = Pinecone(api_key=api_key)
print("Indexes:", [i["name"] for i in pc.list_indexes()])

index = pc.Index(index_name)
stats = index.describe_index_stats()
print("Index stats:", stats)
