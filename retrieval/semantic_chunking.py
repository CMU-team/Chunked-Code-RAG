import json
import argparse
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer, AutoModel
from llama_index.core import SimpleDirectoryReader
from tqdm import tqdm

class StarCoderEmbedding:
    def __init__(self, model_name: str):
        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed(self, text: str):
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(**inputs)
            # Use the last hidden state as embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        return embeddings


def semantic_chunk_text(text: str) -> list[str]:
    # Initialize the HuggingFace embedding model
    embed_model = StarCoderEmbedding(model_name="bigcode/starcoderbase")
    parser = SemanticSplitterNodeParser(embed_model=embed_model)
    
    # Create a list containing the single Document object
    documents = [Document(text=text)]
    
    # Parse the document into nodes
    nodes = parser.get_nodes_from_documents(documents)
    return [node.text for node in nodes]


def main():
    results = [json.loads(l.strip()) for l in open(args.results_path, 'r')]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    proc_results = []
    for r in tqdm(results):
        chunks = semantic_chunk_text(r["text"])
        for c in chunks:
            pd = {"_id": r.get("_id", None), "title": r.get("title", None), "metadata": r.get("metadata", None)}
            pd['text'] = c
            proc_results.append(pd)
    

    output_path = args.results_path.replace(".json", f"_semantic.json")
    print("Output Result Path: ", output_path)
    with open(output_path, 'w') as fw:
        for pr in proc_results:
            fw.write(json.dumps(pr) + '\n')
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="bigcode/starcoder2-7b")

    args = parser.parse_args()

    main()