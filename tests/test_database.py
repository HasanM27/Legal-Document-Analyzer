import chromadb
client = chromadb.PersistentClient(path="C:\Windows D\Study\4thSemetser\AI\Project\legal_assistant\chroma_db")
print(client.list_collections())