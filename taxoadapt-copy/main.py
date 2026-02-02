import os
import json
from collections import deque
from contextlib import redirect_stdout
import argparse
from tqdm import tqdm
import pandas as pd
import time
from datetime import timedelta

from model_definitions import initializeLLM, promptLLM, constructPrompt, token_tracker
from prompts import multi_dim_prompt, NodeListSchema, type_cls_system_instruction, type_cls_main_prompt, TypeClsSchema
from taxonomy import Node, DAG
from expansion import expandNodeWidth, expandNodeDepth
from paper import Paper
from utils import clean_json_string
from json_utils import safe_parse_json

# Optional: datasets library (only needed for non-CSV datasets)
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: 'datasets' library not available. Only CSV input will work.")

def construct_dataset(args):
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    
    if args.dataset == 'naloxone_reddit':
        # Load from CSV file
        print(f"Loading data from CSV: {args.csv_path}")
        df = pd.read_csv(args.csv_path)
        
        # Sample data if sample_size is specified
        if args.sample_size and args.sample_size > 0 and len(df) > args.sample_size:
            print(f"Sampling {args.sample_size} posts from {len(df)} total posts...")
            df = df.sample(n=args.sample_size, random_state=42)
            print(f"Sample selected.")
        else:
            print(f"Using all {len(df)} posts (no sampling)")
        
        internal_collection = {}
        
        with open(os.path.join(args.data_dir, 'internal.txt'), 'w') as i:
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading posts"):
                # Use title if available (for submissions), otherwise create one
                if pd.notna(row.get('title')) and row.get('kind') == 'submissions':
                    title = row['title']
                else:
                    # For comments, create a descriptive title
                    title = f"Comment in r/{row['subreddit']}"
                
                # Text is the main content
                text = str(row.get('text', ''))
                if not text or text == 'nan':
                    continue
                
                temp_dict = {"Title": title, "Abstract": text}
                formatted_dict = json.dumps(temp_dict)
                i.write(f'{formatted_dict}\n')
                
                internal_collection[idx] = Paper(idx, title, text, label_opts=['taxonomy'], internal=True)
        
        internal_count = len(internal_collection)
        print(f"Total # of Reddit Posts: {internal_count}")
        return internal_collection, internal_count
    
    else:
        # Original dataset loading logic
        if not DATASETS_AVAILABLE:
            raise ImportError(
                f"Dataset '{args.dataset}' requires 'datasets' library. "
                "Install with: uv add datasets\n"
                "Or use --dataset naloxone_reddit with CSV input."
            )
        
        split = 'train'
        
        if args.dataset == 'emnlp_2024':
            ds = load_dataset("EMNLP/EMNLP2024-papers")
        elif args.dataset == 'emnlp_2022':
            ds = load_dataset("TimSchopf/nlp_taxonomy_data")
            split = 'test'
        elif args.dataset == 'cvpr_2024':
            ds = load_dataset("DeepNLP/CVPR-2024-Accepted-Papers")
        elif args.dataset == 'cvpr_2020':
            ds = load_dataset("DeepNLP/CVPR-2020-Accepted-Papers")
        elif args.dataset == 'iclr_2024':
            ds = load_dataset("DeepNLP/ICLR-2024-Accepted-Papers")
        elif args.dataset == 'iclr_2021':
            ds = load_dataset("DeepNLP/ICLR-2021-Accepted-Papers")
        elif args.dataset == 'icra_2024':
            ds = load_dataset("DeepNLP/ICRA-2024-Accepted-Papers")
        else:
            ds = load_dataset("DeepNLP/ICRA-2020-Accepted-Papers")
        
        
        internal_collection = {}

        with open(os.path.join(args.data_dir, 'internal.txt'), 'w') as i:
            internal_count = 0
            id = 0
            for p in tqdm(ds[split]):
                if ('title' not in p) and ('abstract' not in p):
                    continue
                
                temp_dict = {"Title": p['title'], "Abstract": p['abstract']}
                formatted_dict = json.dumps(temp_dict)
                i.write(f'{formatted_dict}\n')
                internal_collection[id] = Paper(id, p['title'], p['abstract'], label_opts=['taxonomy'], internal=True)
                internal_count += 1
                id += 1
            print("Total # of Papers: ", internal_count)
        
        return internal_collection, internal_count

def initialize_DAG(args):
    ## we want to make this a directed acyclic graph (DAG) so maintain a list of the nodes
    id2node = {}
    label2node = {}
    
    mod_topic = args.topic.replace(' ', '_').lower()
    mod_full_topic = args.topic.replace(' ', '_').lower() + "_taxonomy"
    root = Node(
            id=0,
            label=mod_topic,
            dimension='taxonomy'
        )
    id2node[0] = root
    label2node[mod_full_topic] = root
    idx = 1

    queue = deque([root])

    while queue:
        curr_node = queue.popleft()
        label = curr_node.label
        # expand
        system_instruction, main_prompt, json_output_format = multi_dim_prompt(curr_node)
        prompts = [constructPrompt(args, system_instruction, main_prompt + "\n\n" + json_output_format)]
        outputs = promptLLM(args=args, prompts=prompts, schema=NodeListSchema, max_new_tokens=3000, json_mode=True, temperature=0.01, top_p=1.0)[0]
        
        # Try to parse JSON output with improved error handling
        outputs = safe_parse_json(outputs, default={}, context=f"Expanding node '{label}'")
        if not outputs:
            print(f"⚠ Skipping node '{label}' due to parsing failure\n")
            continue
        
        outputs = outputs.get('root_topic', outputs.get(label, outputs))

        # add all children
        for key, value in outputs.items():
            mod_key = key.replace(' ', '_').lower()
            mod_full_key = mod_key + "_taxonomy"
            if mod_full_key not in label2node:
                child_node = Node(
                        id=len(id2node),
                        label=mod_key,
                        dimension='taxonomy',
                        description=value['description'],
                        parents=[curr_node]
                    )
                curr_node.add_child(mod_key, child_node)
                id2node[child_node.id] = child_node
                label2node[mod_full_key] = child_node
                if child_node.level < args.init_levels:
                    queue.append(child_node)
            elif label2node[mod_full_key] in label2node[label + "_taxonomy"].get_ancestors():
                continue
            else:
                child_node = label2node[mod_full_key]
                curr_node.add_child(mod_key, child_node)
                child_node.add_parent(curr_node)

    return root, id2node, label2node


def main(args):
    start_time = time.time()
    
    print("######## STEP 1: LOAD IN DATASET ########")

    internal_collection, internal_count = construct_dataset(args)
    
    print(f'Internal: {internal_count}')

    print("######## STEP 2: INITIALIZE DAG ########")
    args = initializeLLM(args)

    root, id2node, label2node = initialize_DAG(args)

    with open(f'{args.data_dir}/initial_taxo.txt', 'w') as f:
        with redirect_stdout(f):
            root.display(0, indent_multiplier=5)

    print("######## STEP 3: ASSIGN ALL POSTS TO ROOT ########")
    
    # Assign all papers to root node
    root.papers = {}
    for p_id, paper in internal_collection.items():
        paper.labels = {'taxonomy': []}
        root.papers[p_id] = paper
    
    print(f"Assigned {len(root.papers)} posts to root node")


    # for each node, classify its papers for the children or perform depth expansion
    print("######## STEP 4: ITERATIVELY CLASSIFY & EXPAND ########")

    visited = set()
    queue = deque([root])

    while queue:
        curr_node = queue.popleft()
        print(f'VISITING {curr_node.label} AT LEVEL {curr_node.level}. WE HAVE {len(queue)} NODES LEFT IN THE QUEUE!')
        
        if len(curr_node.children) > 0:
            if curr_node.id in visited:
                continue
            visited.add(curr_node.id)

            # classify
            curr_node.classify_node(args, label2node, visited)

            # sibling expansion if needed
            new_sibs = expandNodeWidth(args, curr_node, id2node, label2node)
            print(f'(WIDTH EXPANSION) new children for {curr_node.label}: {str((new_sibs))}')

            # re-classify and re-do process if necessary
            if len(new_sibs) > 0:
                curr_node.classify_node(args, label2node, visited)
            
            # add children to queue if constraints are met
            for child_label, child_node in curr_node.children.items():
                c_papers = label2node[child_label + "_taxonomy"].papers
                if (child_node.level < args.max_depth) and (len(c_papers) > args.max_density):
                    queue.append(child_node)
        else:
            # no children -> perform depth expansion
            new_children, success = expandNodeDepth(args, curr_node, id2node, label2node)
            print(f'(DEPTH EXPANSION) new {len(new_children)} children for {curr_node.label}: {str((new_children))}')
            if (len(new_children) > 0) and success:
                queue.append(curr_node)
    
    print("######## STEP 5: SAVE THE TAXONOMY ########")
    with open(f'{args.data_dir}/final_taxonomy.txt', 'w') as f:
        with redirect_stdout(f):
            taxo_dict = root.display(0, indent_multiplier=5)

    with open(f'{args.data_dir}/final_taxonomy.json', 'w', encoding='utf-8') as f:
        json.dump(taxo_dict, f, ensure_ascii=False, indent=4)
    
    # Calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed_time)))
    
    # Print summaries
    print("\n")
    print("=" * 80)
    print("EXECUTION TIME")
    print("=" * 80)
    print(f"Total Time: {elapsed_str} ({elapsed_time:.2f} seconds)")
    print("=" * 80)
    
    token_tracker.print_summary()
    
    # Save statistics to file
    stats = token_tracker.get_summary()
    stats['execution_time_seconds'] = elapsed_time
    stats['execution_time_formatted'] = elapsed_str
    
    with open(f'{args.data_dir}/statistics.json', 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"\nResults saved to: {args.data_dir}/")
    print(f"  - final_taxonomy.txt")
    print(f"  - final_taxonomy.json")
    print(f"  - statistics.json")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=str, default='naloxone discussion')
    parser.add_argument('--dataset', type=str, default='naloxone_reddit')
    parser.add_argument('--csv_path', type=str, default='naloxone_mentions.csv')
    parser.add_argument('--sample_size', type=int, default=None, 
                        help='Number of posts to sample. Use 0 or None for all data.')
    parser.add_argument('--llm', type=str, default='custom')
    parser.add_argument('--max_depth', type=int, default=2)
    parser.add_argument('--init_levels', type=int, default=1)
    parser.add_argument('--max_density', type=int, default=40)
    args = parser.parse_args()

    args.data_dir = f"datasets/{args.dataset.lower().replace(' ', '_')}"
    args.internal = f"{args.dataset}.txt"

    main(args)