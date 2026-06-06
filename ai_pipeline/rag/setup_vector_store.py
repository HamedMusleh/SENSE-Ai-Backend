from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create Vector Store
vector_store = client.vector_stores.create(
    name="sense_knowledge_base"
)

print("Vector Store Created:")
print(vector_store.id)

# Files to upload
files_to_upload = [
    "resources/annotation_guide.md",
    "resources/safety_rules.md",
    "resources/triage_labels.md",
    "resources/system_scope.md",
    "resources/dataset_examples.jsonl",
]

# Upload files
for path in files_to_upload:

    print(f"Uploading: {path}")

    uploaded_file = client.files.create(
        file=open(path, "rb"),
        purpose="assistants"
    )

    client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=uploaded_file.id
    )

print("\nDONE")
print("SAVE THIS VECTOR STORE ID:")
print(vector_store.id)