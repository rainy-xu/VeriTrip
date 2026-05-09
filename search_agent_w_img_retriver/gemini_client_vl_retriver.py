import os
import json
import time
from datetime import datetime,timedelta
import argparse
from pathlib import Path
import csv
import sys
from tqdm import tqdm
from .prompts import format_query,get_search_system_prompt
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


import asyncio
from concurrent.futures import ThreadPoolExecutor
from rich import print as rprint
import openai
from google.genai import types

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from searcher.searchers import SearcherType
from searcher.restaurant import Restaurants
from .utils import extract_retrieved_docids_from_result
from transformers import AutoTokenizer
from typing import List, Any, Dict
from searcher.image_searcher import ImageIndexingManager

from .tool_handler import SearchToolHandler, RestaurantToolHandler, ImageToolHandler, CombinedToolHandler
from PIL import Image

import base64
# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def compress_image_to_base64(image_path, max_size=(1024, 1024), quality=85):
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.thumbnail(max_size, Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
def build_request_with_img_list(
    query: str,
    img_list: list,
    model: str,
    max_tokens: int,
    tool_handler: 'CombinedToolHandler',
    system_prompt: bool,
    reasoning_effort: str | None = None,
    query_template: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
):

    system_instruction = None
    if system_prompt:
        sys_text = get_search_system_prompt('en')
        system_instruction = {
            "parts": [{"text": sys_text}]
        }
        
    formatted_query = format_query(query, query_template)
    

    user_parts = [{"text": formatted_query}]
    
    for img_path in img_list:
        img_item_list = img_path.split('|')
        for i in img_item_list:
            if not i.strip(): continue
            try:

                base64_image = compress_image_to_base64(i)
                
                user_parts.append({"text": f"Image source: {i}"})
                user_parts.append({
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": base64_image 
                    }
                })
            except Exception as e:
                # breakpoint()
                print(f"Error encoding image {i}: {e}")

    contents = [
        {
            "role": "user",
            "parts": user_parts
        }
    ]
    
    openai_tools = tool_handler.get_tool_definitions()
    gemini_tools = []
    if openai_tools:
        funcs = [t['function'] for t in openai_tools if 'function' in t]
        if funcs:
            gemini_tools = [{"functionDeclarations": funcs}]

    generation_config = {
        "maxOutputTokens": max_tokens,
        "temperature": temperature if temperature is not None else 0.0,
        "topP": top_p if top_p is not None else 0.95,
    }

    body = {
        "contents": contents,
        "tools": gemini_tools,
        "generationConfig": generation_config,
    }
    
    if system_instruction:
        body["systemInstruction"] = system_instruction


    return body


executor = ThreadPoolExecutor(max_workers=10)

async def run_conversation_with_tools_async(
    client: "openai.AsyncOpenAI", 
    initial_request: dict, 
    tool_handler: 'CombinedToolHandler', 
    max_iterations: int = 5
):
    contents = initial_request["contents"].copy()
    global_max_tokens = initial_request['generationConfig']['maxOutputTokens']
    
    cumulative_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
    }
    
    combined_output = []
    tool_outputs = []
    all_responses = []
    visual_tool_names = ["search_similar_images_by_path"]
    
    loop = asyncio.get_running_loop()

    for iteration in range(max_iterations):
        try:
            current_request = initial_request.copy()
            current_request["contents"] = contents
            
            response = await client.chat.completions.create(current_request)
            all_responses.append(response)
        except Exception as e:
            print(f"API Error in iteration {iteration}: {e}")
            break

        if 'usageMetadata' in response:
            usage = response['usageMetadata']
            cumulative_usage["input_tokens"] = usage.get('promptTokenCount', 0)
            cumulative_usage["output_tokens"] = usage.get('candidatesTokenCount', 0)
            cumulative_usage["total_tokens"] = usage.get('totalTokenCount', 0)


        if 'candidates' not in response or not response['candidates']:
            break
            
        candidate = response['candidates'][0]
        content = candidate.get('content', {})
        parts = content.get('parts', [])
        
        if content:
            contents.append(content)
            combined_output.append(content)


        function_calls = [p['functionCall'] for p in parts if 'functionCall' in p]
        
        if not function_calls:

            return response, combined_output, cumulative_usage, tool_outputs


        async def _exec_tool(fc):
            name = fc['name']
            args = fc.get('args', {})
            try:
                res = await loop.run_in_executor(executor, tool_handler.execute_tool, name, args)
                if res is None: res = "Success"
            except Exception as e:
                res = str(e)
            return name, args, res


        tool_tasks = [asyncio.create_task(_exec_tool(fc)) for fc in function_calls]
        tool_results = await asyncio.gather(*tool_tasks)

        new_retrieved_images = []
        function_response_parts = []

        for name, args, result in tool_results:

            tool_outputs.append({
                "tool_name": name,
                "arguments": args,
                "output": result,
                "status": "completed"
            })
            
            function_response_parts.append({
                "functionResponse": {
                    "name": name,
                    "response": {"result": result}
                }
            })

            if name in visual_tool_names:
                try:
                    search_results = json.loads(result) if isinstance(result, str) else result
                    if isinstance(search_results, list):
                        for item in search_results:
                            if isinstance(item, dict) and "file_path" in item:
                                new_retrieved_images.append(item["file_path"])
                except Exception as e:
                    print(f"Error parsing visual tool output: {e}")


        if function_response_parts:
            contents.append({
                "role": "function",
                "parts": function_response_parts
            })


        if new_retrieved_images:
            img_tasks = [
                loop.run_in_executor(executor, compress_image_to_base64, path) 
                for path in new_retrieved_images
            ]
            base64_list = await asyncio.gather(*img_tasks, return_exceptions=True)

            image_content_parts = [{"text": "I have retrieved the following images based on your search:"}]
            
            for path, b64 in zip(new_retrieved_images, base64_list):
                if isinstance(b64, Exception):
                    print(f"Error processing image {path}: {b64}")
                    continue
                
                image_content_parts.append({
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": b64
                    }
                })
            
            contents.append({
                "role": "user",
                "parts": image_content_parts
            })
            print(f"Iteration {iteration}: Concurrently processed {len(new_retrieved_images)} images.")

    final_response = all_responses[-1] if all_responses else None
    return final_response, combined_output, cumulative_usage, tool_outputs





def _persist_response(out_dir: str, request_body: dict, response, combined_output=None, cumulative_usage=None, tool_outputs: dict = None, *, query_id: str | None = None):
    """
    Persist request & response JSON for later inspection.
    This version handles tool calls without IDs by assuming sequential order.
    """
    
    os.makedirs(out_dir, exist_ok=True)


    if combined_output and hasattr(combined_output[0], 'model_dump'):
        combined_output_dicts = [item.model_dump(mode="json", exclude_none=True) for item in combined_output]
    else:
        combined_output_dicts = combined_output or []

    normalized_results: list[dict] = []
    tool_call_counts: dict[str, int] = {}

    tool_output_index = 0

    for message_dict in combined_output_dicts:

        tool_calls = message_dict.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict): continue

                function_details = tool_call.get("function", {})
                tool_name = function_details.get("name")
                
                if not tool_name:
                    continue 


                tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                

                tool_output_content = "Error: Tool output not found or index out of bounds."
                if tool_output_index < len(tool_outputs):

                    current_tool_output = tool_outputs[tool_output_index]
                    tool_output_content = current_tool_output.get("output", "Error: 'output' key missing in tool result.")
                    tool_output_index += 1
                
                arguments = function_details.get("arguments")
                try:
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                except (json.JSONDecodeError, TypeError):
                    pass

                normalized_results.append({
                    "type": "tool_call",
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "output": tool_output_content,
                })

        content = message_dict.get("content")
        
        if isinstance(content, list):
            final_text_output = ""
            for part in content:
                if isinstance(part, dict) and 'text' in part:
                    text_content = part['text']
                    if text_content.strip().startswith("#transportationTable"):
                        final_text_output += text_content + "\n"
            
            if final_text_output.strip():
                normalized_results.append({
                    "type": "output_text",
                    "tool_name": None,
                    "arguments": None,
                    "output": final_text_output.strip(),
                })
        
        elif isinstance(content, str) and content.strip():

            if content.strip().startswith("#transportationTable"):
                normalized_results.append({
                    "type": "output_text",
                    "tool_name": None,
                    "arguments": None,
                    "output": content,
                })
    
    normalized_usage = {
        "input_tokens": cumulative_usage["input_tokens"],
        "input_tokens_cached": cumulative_usage["input_tokens_details"]["cached_tokens"],
        "output_tokens": cumulative_usage["output_tokens"],
        "included_reasoning_tokens": cumulative_usage["output_tokens_details"]["reasoning_tokens"],
        "total_tokens": cumulative_usage["total_tokens"],
    }

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    filename = os.path.join(out_dir, f"run_{ts}.json")
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "model": request_body.get("model"),
                "reasoning": request_body.get("reasoning"),
                "output_dir": str(out_dir),
                "max_tokens": request_body.get("max_tokens"),
            },
            "query_id": query_id,
            "tool_call_counts": tool_call_counts,
            "usage": normalized_usage,
            "status": response.choices[0].finish_reason,
            "retrieved_docids": extract_retrieved_docids_from_result(normalized_results),
            "result": normalized_results,
        }, f, indent=2, ensure_ascii=False,default=str)
    
    print("Saved response to", filename, "| tool call counts:", tool_call_counts)


def _process_csv_dataset(csv_path: str, client: openai.OpenAI, args, tool_handler: CombinedToolHandler):
    """Process a CSV file of (id \t query) pairs sequentially and persist responses."""
    # breakpoint()
    dataset_path = Path(csv_path)
    out_dir = Path(args.output_dir).expanduser().resolve()

    queries: list[tuple[str, str, list]] = []
    
    if not dataset_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # MODIFIED: Use 'utf-8-sig' to handle BOM and specify proper CSV dialect
    with dataset_path.open(newline="", encoding="utf-8-sig") as f:
        # DictReader automatically uses the first row as headers
        # Added quoting parameter to handle fields with special characters
        reader = csv.DictReader(f, quoting=csv.QUOTE_MINIMAL)
        
        # Verify required headers exist
        if reader.fieldnames is None:
            raise ValueError(f"CSV file '{csv_path}' appears to be empty or malformed")
            
        required_headers = {'id', 'nature_language_img'}
        actual_headers = set(reader.fieldnames)
        
        if not required_headers.issubset(actual_headers):
            missing = required_headers - actual_headers
            raise ValueError(f"CSV file '{csv_path}' is missing required columns: {missing}")

        for line_num, row in enumerate(reader, start=2):  # start=2 because line 1 is header

            # Access columns by name
            query_id = row.get('id', '').strip()
            query_text = row.get('nature_language_img', '').strip()
            
            # Handle image paths
            attraction_images_str = row.get('p_attraction_img', '').strip()

            attraction_images = attraction_images_str.split("|") if attraction_images_str else []
            attraction_images = [img.strip() for img in attraction_images if img.strip()]

            image_list = attraction_images
            
            # Skip if no images
            if len(image_list) == 0:
                print(f"Line {line_num}: Skipping query_id={query_id} - no images")
                continue
            
            # Validate required fields
            if not query_id or not query_text:
                print(f"Warning: Line {line_num} - Skipping row with missing id or nature_language: {row}")
                continue
            
            queries.append((query_id, query_text, image_list))

    processed_ids = set()
    if out_dir.exists():
        for json_path in out_dir.glob("run_*.json"):
            try:
                with json_path.open("r", encoding="utf-8") as jf:
                    meta = json.load(jf)
                    qid_saved = meta.get("query_id")
                    if qid_saved:
                        processed_ids.add(str(qid_saved))
            except Exception:
                continue  # ignore corrupt files
    
    remaining = [(qid, qtext, img_list) for qid, qtext, img_list in queries if qid not in processed_ids]
    
    print(
        f"Processing {len(remaining)} remaining queries (skipping {len(processed_ids)}) from {dataset_path} ..."
    )
    
    import threading
    completed_lock = threading.Lock()
    completed_count = [0]

    query_counter = 0
    PAUSE_THRESHOLD = 10  
    PAUSE_DURATION = 3600 

    def _handle_single_query(qid: str, qtext: str, qimgs: list[str], pbar=None):
        request_body = build_request_with_img_list(
            query=qtext,
            img_list=qimgs,
            model=args.model,
            max_tokens=args.max_tokens,
            tool_handler=tool_handler,
            system_prompt=args.system,
            reasoning_effort=args.reasoning_effort,
            query_template=args.query_template,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        if True:
            response, combined_output, cumulative_usage, tool_outputs = await run_conversation_with_tools_async(
                client, request_body, tool_handler, args.max_iterations
            )
            # breakpoint()
            if response.choices[0].finish_reason == "stop":
                with completed_lock:
                    completed_count[0] += 1
                    if pbar:
                        pbar.set_postfix(completed=completed_count[0])
            
            _persist_response(out_dir, request_body, response, combined_output, cumulative_usage, tool_outputs, query_id=qid)
            
    
    if args.num_threads <= 1:

        with tqdm(remaining, desc="Queries", unit="query") as pbar:
            for qid, qtext, qimgs in pbar:
                _handle_single_query(qid, qtext, qimgs, pbar)
                time.sleep(20)
    else:

        with ThreadPoolExecutor(max_workers=args.num_threads) as executor, \
            tqdm(total=len(remaining), desc="Queries", unit="query") as pbar:
            
            futures = {executor.submit(_handle_single_query, qid, qtext, qimgs, pbar): qid for qid, qtext, qimgs in remaining}

            for future in as_completed(futures):
                pbar.update(1)
                try:
                    future.result() 
                except Exception as exc:
                    print(f"\nQuery id={futures[future]} generated an exception: {exc}")



def main():
    parser = argparse.ArgumentParser(description="Call OpenAI Responses API with native function calling and local search.")
    parser.add_argument("--query", default=os.environ.get("QUERY_CSV_PATH", ""), help="Path to query CSV file.")
    parser.add_argument("--caption", default=os.environ.get("CAPTION_SAVE_PATH", ""), help="Path to caption JSON file.")
    parser.add_argument("--model", default="gpt-4.1", help="Model name (default: %(default)s)")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Max tokens to generate (default: %(default)s)")
    parser.add_argument("--system", default=True, help="Optional system prompt")
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Optional reasoning effort level (for reasoning-capable models)",
    )
    parser.add_argument("--output-dir", default="runs/bm25/gpt_4_1", help="Directory to store logs (default: %(default)s)")
    parser.add_argument("--query-template", choices=["QUERY_TEMPLATE", "QUERY_TEMPLATE_NO_GET_DOCUMENT", "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION"], default="QUERY_TEMPLATE", help="Specify the query template to use (default: %(default)s)")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for the model (default: use model defaults)")
    parser.add_argument("--top_p", type=float, default=None, help="Top P for the model (default: use model defaults)")
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of parallel threads for dataset processing (default: %(default)s)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts per query on transient errors (default: %(default)s)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum number of conversation rounds with function calls (default: %(default)s)",
    )
    
    parser.add_argument(
        "--searcher-type",
        choices=SearcherType.get_choices(),
        required=True,
        help=f"Type of searcher to use: {', '.join(SearcherType.get_choices())}",
    )
    
    # Server configuration arguments
    parser.add_argument(
        "--snippet-max-tokens",
        type=int,
        default=512,
        help="Number of tokens to include for each document snippet in search results using Qwen/Qwen3-0.6B tokenizer (default: 512).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Fixed number of search results to return for all queries in this session (default: 5).",
    )
    parser.add_argument(
        "--get-document",
        default="True",
        help="If set, register both the search tool and the get_document tool.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face token for accessing private datasets/models. If not provided, will use environment variables or CLI login.",
    )
    parser.add_argument(
        "--hf-home",
        type=str,
        help="Hugging Face home directory for caching models and datasets. If not provided, will use environment variables or default.",
    )
    
    temp_args, _ = parser.parse_known_args()
    searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
    searcher_class.parse_args(parser)

    args = parser.parse_args()
    
    if args.hf_token:
        print(f"[DEBUG] Setting HF token from CLI argument: {args.hf_token[:10]}...")
        os.environ['HF_TOKEN'] = args.hf_token
        os.environ['HUGGINGFACE_HUB_TOKEN'] = args.hf_token
    
    if args.hf_home:
        print(f"[DEBUG] Setting HF home from CLI argument: {args.hf_home}")
        os.environ['HF_HOME'] = args.hf_home
    

    client = openai.AsyncOpenAI(
        api_key=os.environ.get("LLM_API_KEY", ""),
        base_url=os.environ.get("LLM_API_BASE_URL", "")
    )

    searcher = searcher_class(args)
    
    tool_handler = SearchToolHandler(
        searcher=searcher,
        snippet_max_tokens=args.snippet_max_tokens,
        k=args.k,
        include_get_document=args.get_document
    )
    restaurant_manager = Restaurants()
    restaurant_handler = RestaurantToolHandler(restaurant_manager=restaurant_manager)
    image_manager = ImageIndexingManager(
        root_dir=os.environ.get("IMAGE_ROOT_DIR_VARI", ""),
        index_path=os.environ.get("VARI_IMAGE_INDEX_PATH", ""),
        embedding_path=os.environ.get("VARI_IMAGE_EMBEDDING_PATH", ""))
    image_handler = ImageToolHandler(image_manager)
    all_handlers = [tool_handler, restaurant_handler,image_handler]
    combined_handler = CombinedToolHandler(handlers=all_handlers)


    tools_registered = ["travel_search"]
    if args.get_document:
        tools_registered.append("get_document")
    

    tools_str = ", ".join(tools_registered)
    
    print(
        f"Search agent started with {searcher.search_type} search (snippet_max_tokens={args.snippet_max_tokens}, k={args.k})"
    )
    print(f"Registered tools: {tools_str}")
    # breakpoint()
    if isinstance(args.query, str):

        qstr = args.query.strip()

        # caption = json.load(open(args.caption, 'r'))
        if qstr.lower().endswith(".csv"):
            potential_path = Path(qstr)
            # try:
            if True:
                if potential_path.is_file():
                    _process_csv_dataset(str(potential_path), client, args, combined_handler)
                    return
            # except OSError:
            #     pass



if __name__ == "__main__":
    main()
