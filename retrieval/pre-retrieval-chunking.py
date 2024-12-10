"""Post-process docs before generation.
- solutions: none
- tutorials: first 200 tokens, first 500 tokens, none
"""

import json
import argparse
from transformers import AutoTokenizer

def get_chunks(text: str, num_tokens: int, tokenizer: AutoTokenizer) -> list[str]:
    # aggregate content line by line
    line_tokens = [tokenizer.tokenize(line) for line in text.split('\n')]

    # chunk every `num_tokens` tokens
    token_chunks = []
    curr_tokens, curr_chunk = 0, []
    for lt in line_tokens:
        if len(lt) + curr_tokens > num_tokens:
            token_chunks.append(curr_chunk)
            curr_tokens, curr_chunk = 0, []
        else:
            curr_chunk.append(lt)
            curr_tokens += len(lt)
    if curr_tokens > 0:
        token_chunks.append(curr_chunk)
    
    # convert tokens back to text strings
    text_chunks = []
    for ck in token_chunks:
        ck = [tokenizer.convert_tokens_to_string(lt) for lt in ck]
        ck = '\n'.join(ck)
        text_chunks.append(ck)
    return text_chunks



def truncate_text(text: str, num_tokens: int, tokenizer: AutoTokenizer):
    
    return get_chunks(text, num_tokens, tokenizer)


def main():
    results = [json.loads(l.strip()) for l in open(args.results_path, 'r')]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    proc_results = []
    for r in results:
        chunks = truncate_text(r["text"], args.max_num_tokens, tokenizer)
        for c in chunks:
            pd = {"_id": r.get("_id", None), "title": r.get("title", None), "metadata": r.get("metadata", None)}
            pd['text'] = c
            proc_results.append(pd)
    

    output_path = args.results_path.replace(".json", f"_{args.max_num_tokens}-tokens.json")
    print("Output Result Path: ", output_path)
    with open(output_path, 'w') as fw:
        for pr in proc_results:
            fw.write(json.dumps(pr) + '\n')
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="bigcode/starcoder2-7b")
    parser.add_argument("--max_num_tokens", type=int, default=None)
    parser.add_argument("--is_docs", action="store_true")
    args = parser.parse_args()

    main()
