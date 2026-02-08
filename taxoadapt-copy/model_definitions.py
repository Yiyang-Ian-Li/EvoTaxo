# Core imports
import os
import json
import requests
import numpy as np
from tqdm import tqdm

# OpenAI imports
import openai
from openai import OpenAI

# GPU-related imports (optional, only needed for vllm mode)
try:
    from transformers import pipeline
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    import torch
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vllm/transformers not available. Only custom/gpt modes will work.")

openai_key = os.getenv('OPENAI_API_KEY')

# Global token usage tracker
class TokenUsageTracker:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
    
    def add_usage(self, prompt_tokens, completion_tokens):
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.call_count += 1
    
    def get_summary(self):
        return {
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens,
            'total_tokens': self.total_tokens,
            'api_calls': self.call_count
        }
    
    def print_summary(self):
        print("\n" + "="*80)
        print("TOKEN USAGE SUMMARY")
        print("="*80)
        print(f"Total API Calls: {self.call_count:,}")
        print(f"Prompt Tokens: {self.total_prompt_tokens:,}")
        print(f"Completion Tokens: {self.total_completion_tokens:,}")
        print(f"Total Tokens: {self.total_tokens:,}")
        print("="*80 + "\n")

token_tracker = TokenUsageTracker()


# map each term in text to word_id
def get_vocab_idx(split_text: str, tok_lens):

	vocab_idx = {}
	start = 0

	for w in split_text:
		# print(w, start, start + len(tok_lens[w]))
		if w not in vocab_idx:
			vocab_idx[w] = []

		vocab_idx[w].extend(np.arange(start, start + len(tok_lens[w])))

		start += len(tok_lens[w])

	return vocab_idx

def get_hidden_states(encoded, data_idx, model, layers, static_emb):
	"""Push input IDs through model. Stack and sum `layers` (last four by default).
	Select only those subword token outputs that belong to our word of interest
	and average them."""
	with torch.no_grad():
		output = model(**encoded)

	# Get all hidden states
	states = output.hidden_states
	# Stack and sum all requested layers
	output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

	# Only select the tokens that constitute the requested word

	for w in data_idx:
		static_emb[w] += output[data_idx[w]].sum(dim=0).cpu().numpy()

def chunkify(text, token_lens, length=512):
	chunks = [[]]
	split_text = text.split()
	count = 0
	for word in split_text:
		new_count = count + len(token_lens[word]) + 2 # 2 for [CLS] and [SEP]
		if new_count > length:
			chunks.append([word])
			count = len(token_lens[word])
		else:
			chunks[len(chunks) - 1].append(word)
			count = new_count
	
	return chunks

def constructPrompt(args, init_prompt, main_prompt):
	if (args.llm == 'gpt'):
		return [
            {"role": "system", "content": init_prompt},
            {"role": "user", "content": main_prompt}]
	else:
		return init_prompt + "\n\n" + main_prompt

def initializeLLM(args):
	args.client = {}
	
	if args.llm == 'vllm':
		if not VLLM_AVAILABLE:
			raise ImportError(
				"vllm mode requires transformers, vllm, and torch. "
				"Install with: uv add transformers vllm torch"
			)
		args.client['vllm'] = LLM(
			model="meta-llama/Meta-Llama-3.1-8B-Instruct",
			tensor_parallel_size=4,
			gpu_memory_utilization=0.95,
			max_num_batched_tokens=4096,
			max_num_seqs=1000,
			enable_prefix_caching=True
		)
	elif args.llm == 'custom':
		# Custom API setup
		args.api_url = "https://openwebui.crc.nd.edu/api/v1/chat/completions"
		args.api_key = 'sk-5aa5ea0263c942f896210d035529b47b'
		args.model_name = "gpt-oss:120b"
	elif args.llm == 'gpt':
		args.client[args.llm] = OpenAI(api_key=openai_key)
	
	return args

def promptCustomAPI(args, prompts, schema=None, max_new_tokens=1024, json_mode=True, temperature=0.1, top_p=0.99):
	outputs = []
	headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
	
	for messages in tqdm(prompts, desc="API Calls"):
		# Convert messages format if needed
		if isinstance(messages, str):
			messages_list = [{"role": "user", "content": messages}]
		else:
			messages_list = messages
		
		payload = {
			"model": args.model_name,
			"messages": messages_list,
			"temperature": temperature,
			"top_p": top_p,
			"max_tokens": max_new_tokens
		}
		
		try:
			response = requests.post(args.api_url, headers=headers, data=json.dumps(payload), timeout=300)
			response.raise_for_status()
			result = response.json()
			
			# Extract content
			content = result['choices'][0]['message']['content']
			outputs.append(content)
			
			# Track token usage
			if 'usage' in result:
				token_tracker.add_usage(
					result['usage'].get('prompt_tokens', 0),
					result['usage'].get('completion_tokens', 0)
				)
				
		except requests.exceptions.Timeout:
			print(f"⚠ API call timeout (300s). Retrying with empty response...")
			outputs.append('{}')
		except requests.exceptions.RequestException as e:
			print(f"⚠ API request failed: {e}")
			outputs.append('{}')
		except (KeyError, IndexError) as e:
			print(f"⚠ Unexpected API response format: {e}")
			if 'result' in locals():
				print(f"Response: {result}")
			outputs.append('{}')
		except Exception as e:
			print(f"⚠ Unexpected error during API call: {e}")
			outputs.append("{}")
			
	return outputs

def promptGPT(args, prompts, schema=None, max_new_tokens=1024, json_mode=True, temperature=0.1, top_p=0.99):
	outputs = []
	for messages in tqdm(prompts):
		if json_mode:
			response = args.client['gpt'].chat.completions.create(model='gpt-4o-mini-2024-07-18', stream=False, messages=messages, 
									response_format={"type": "json_object"}, temperature=temperature, top_p=top_p, 
									max_tokens=max_new_tokens)
		else:
			response = args.client['gpt'].chat.completions.create(model='gpt-4o-mini-2024-07-18', stream=False, messages=messages, 
							 temperature=temperature, top_p=top_p,
							 max_tokens=max_new_tokens)
		outputs.append(response.choices[0].message.content)
		
		# Track token usage for OpenAI too
		if hasattr(response, 'usage'):
			token_tracker.add_usage(
				response.usage.prompt_tokens,
				response.usage.completion_tokens
			)
	return outputs

def promptLlamaVLLM(args, prompts, schema=None, max_new_tokens=1024, temperature=0.1, top_p=0.99):
    if not VLLM_AVAILABLE:
        raise ImportError("vllm is not available. Use --llm custom or --llm gpt instead.")
    
    if schema is None:
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
    else:
        guided_decoding_params = GuidedDecodingParams(json=schema.model_json_schema())
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, 
                                    guided_decoding=guided_decoding_params)
    generations = args.client['vllm'].generate(prompts, sampling_params)
    
    outputs = []
    for gen in generations:
        outputs.append(gen.outputs[0].text)

    return outputs

def promptLLM(args, prompts, schema=None, max_new_tokens=1024, json_mode=True, temperature=0.1, top_p=0.99):
	if args.llm == 'custom':
		return promptCustomAPI(args, prompts, schema, max_new_tokens, json_mode, temperature, top_p)
	elif args.llm == 'gpt':
		return promptGPT(args, prompts, schema, max_new_tokens, json_mode, temperature, top_p)
	else:
		return promptLlamaVLLM(args, prompts, schema, max_new_tokens, temperature, top_p)
	