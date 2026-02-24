from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Read the content of formula.txt

text = Path("formula.txt").read_text(encoding="utf-8")

# Split text into chunks based on "### topic:"

chunks = []
current_chunk = []

for line in text.splitlines():
    if line.startswith("### topic:"):
        if current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
    current_chunk.append(line)

if current_chunk:
    chunks.append("\n".join(current_chunk))

# Document list to hold processed chunks

documents = []


for chunk in chunks:
    if not chunk.strip():
        continue  # skip empty chunks

    lines = [l for l in chunk.splitlines() if l.strip()]
    if not lines:
        continue

    first_line = lines[0]

    if "topic:" not in first_line.lower():
        continue  # safety check

    topic = (
        first_line
        .split("topic:", 1)[1]
        .strip()
        .strip('"')
    )

    documents.append(
        Document(
            page_content=chunk,
            metadata={
                "topic": topic,
                "source": "formulas.txt"
            }
        )
    )
# print(f"Total documents created: {len(documents)}")

# Initialize HuggingFace Embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Create Chroma vector store from documents

db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="chroma_stores/aptitude_formulas",
    collection_name="aptitude_formulas"
)
