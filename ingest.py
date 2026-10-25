import hashlib
import time
from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vectorstore import vector_store


loader = DirectoryLoader("./data" , glob="**/*.pdf",loader_cls=PyPDFLoader)
docs=loader.load()
print(len(docs))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

all_chunks = text_splitter.split_documents(docs)
print(len(all_chunks))
print(all_chunks[0])
print("checking for new data")
for chunk in all_chunks:
    content_hash = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()
    source = chunk.metadata.get("source","unkown_doc")
    chunk.metadata["id"] = f"{source}_{content_hash}"

existing_id = set(vector_store.get()["ids"])
new_chunks = [c for c in all_chunks if c.metadata["id"] not in existing_id]

print(f"Total chunks found: {len(all_chunks)}")
print(f"Chunks already in DB: {len(existing_id)}")
print(f"New chunks to embed: {len(new_chunks)}")


batch_size = 95

if len(new_chunks)>0:
    for i in range(0,len(new_chunks) , batch_size):
        batch = new_chunks[i:i+batch_size] 
        batch_ids = [doc.metadata["id"] for doc in batch]

        print(f"\nembedding batch {i} to {i+batch_size}")

        vector_store.add_documents(documents=batch , ids=batch_ids)

        if i + batch_size < len(new_chunks):
            print("\nresting for 60 seconds for api to reset")
            time.sleep(60)

    print("\n done syncing all new data")
else: 
    print("\n no new data to embed")

