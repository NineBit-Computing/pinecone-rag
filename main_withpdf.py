# Import the Pinecone library
from pinecone import Pinecone as Pinecone
from pinecone import ServerlessSpec
import time
from PyPDF2 import PdfReader
import cohere

co = cohere.Client('A30oAcq9woG1QXsN2ZWE5oNEUSPjOy3lLqXFK7bK')

# Initialize a Pinecone client with your API key
pc = Pinecone(api_key="pcsk_4vXZfN_PurThYw1uZ4R3jATY2BGkpmHRYqTRmDkCMKxSZ9t9gThq8qdoGsHW5ucNY4rbRX")

pdf_paths = [
    "/home/khushi/pinecone_only/geo1.pdf",  
    "/home/khushi/pinecone_only/geo2.pdf", 
    "/home/khushi/pinecone_only/history1.pdf",
    "/home/khushi/pinecone_only/history2.pdf",
    "/home/khushi/pinecone_only/civics1.pdf",
]

data = []

# Function to process each PDF file
def process_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    file_data = []
    
    # Extract and structure the data
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text.strip():  # Skip empty pages
            file_data.append({"id": f"{pdf_path}_vec{page_number}", "text": text.strip()})
    
    return file_data

# Loop through each PDF file path and process it
for pdf_path in pdf_paths:
    pdf_data = process_pdf(pdf_path)
    data.extend(pdf_data)  # Add the data from this PDF to the main data list

# Output the structured data
#for item in data:
#    print(item)

# Convert the text into numerical vectors that Pinecone can index
embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[d['text'] for d in data],
    parameters={"input_type": "passage", "truncate": "END"}
)

#print(embeddings)

# Create a serverless index
index_name = "example-index"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    ) 

# Wait for the index to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)
# Target the index where you'll store the vector embeddings
index = pc.Index("example-index")

# Prepare the records for upsert
# Each contains an 'id', the embedding 'values', and the original text as 'metadata'
records = []
for d, e in zip(data, embeddings):
    records.append({
        "id": d['id'],
        "values": e['values'],
        "metadata": {'text': d['text']}
    })

# Upsert the records into the index
index.upsert(
    vectors=records,
    namespace="example-namespace"
)
time.sleep(10)  # Wait for the upserted vectors to be indexed

#print(index.describe_index_stats())
def query_chunk(query_text):
# Convert the query into a numerical vector that Pinecone can search with
    query_embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query_text],
        parameters={
            "input_type": "query"
        }
    ) 

# Search the index for the three most similar vectors
    results = index.query(
        namespace="example-namespace",
        vector=query_embedding[0].values,
        top_k=3,
        include_values=False,
        include_metadata=True
    )

#print(results)
    print("Search Results:")
    for match in results["matches"]:
        print(f"Score: {match['score']}, Text: {match['metadata']['text']}")

    return query_text, results 


def generate_answer(query_text, context):
    try:
        response = co.generate(
            model='command-xlarge-nightly',
            prompt = f"Use the information provided below to answer the questions at the end. If the answer to the question is not contained in the provided information, say The answer is not in the context.Context information:{context} Question: {query_text}",
            #prompt=f"Question: {query_text}\n\nContext: {context}\n\nAnswer:",
            max_tokens=150,
            temperature=0.7
        )
        answer = response.generations[0].text.replace("\n", " ").strip()
        print(f"Answer: {answer}")
        return answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return None

while True:
    query_text = input("Enter your question here or print exit to Exit.... \n")

    if query_text.lower() == 'exit':
        print("Exiting.......!")
        break 

    results, content = query_chunk(query_text)

    generate_answer(query_text, content)
