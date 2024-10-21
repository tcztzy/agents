from pathlib import Path
from swarmx import Swarm, Agent

import chromadb

client = chromadb.PersistentClient(str(Path.home() / ".your-agent" / "chroma"))
collection = client.get_or_create_collection(name="your-collection")


def retrieve(query: str) -> str:
    """Retrieve documents from the Database using a query text."""
    return collection.query(query_texts=query, n_results=1)["documents"][0][0]


client = Swarm()

retrieval = Agent(
    name="retrieval",
    instructions="Retrieve documents from the Database using a query text.",
    functions=[retrieve]
)

client.run(retrieval, [{"content": "Can you explain how does openai o1 works?", "role": "user"}])
