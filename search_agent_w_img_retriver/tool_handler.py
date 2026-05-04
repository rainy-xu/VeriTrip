import os
import json
import time
import logging
from datetime import datetime,timedelta
import argparse
from pathlib import Path
import csv
import sys
from tqdm import tqdm
from typing import List, Any, Dict

from transformers import AutoTokenizer


class SearchToolHandler:
    def __init__(self, searcher, snippet_max_tokens: int | None = None, k: int = 5, include_get_document: bool = True):
        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.include_get_document = include_get_document
        
        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            self.tokenizer = AutoTokenizer.from_pretrained(os.environ.get("EMBEDDING_MODEL_PATH", ""))
    
    def get_tool_definitions(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "travel_search",
                    "description": self.searcher.search_description(self.k),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query string"
                            }
                        },
                    },
                    "required": ["query"]
                }
            }
        ]
        
        if self.include_get_document:
            tools.append({
                "type": "function",
                "function":{
                    "name": "get_document",
                    "description": self.searcher.get_document_description(),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "docid": {
                                "type": "string",
                                "description": "Document ID to retrieve"
                            }
                        },
                    },
                    "required": ["docid"],
                }
            })
        
        return tools
    
    def execute_tool(self, tool_name: str, arguments: dict):
        if tool_name == "travel_search":
            return self._search(arguments["query"])
        elif tool_name == "get_document":
            return self._get_document(arguments["docid"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def _search(self, query: str):
        candidates = self.searcher.search(query, self.k)
        
        if self.snippet_max_tokens and self.snippet_max_tokens > 0 and self.tokenizer:
            for cand in candidates:
                text = cand["text"]
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > self.snippet_max_tokens:
                    truncated_tokens = tokens[:self.snippet_max_tokens]
                    cand["snippet"] = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                else:
                    cand["snippet"] = text
        else:
            for cand in candidates:
                cand["snippet"] = cand["text"]
        
        results = []
        for cand in candidates:
            if cand.get("score") is None:
                results.append({
                    "docid": cand["docid"],
                    "snippet": cand["snippet"]
                })
            else:
                results.append({
                    "docid": cand["docid"],
                    "score": cand["score"],
                    "snippet": cand["snippet"]
                })
        
        return json.dumps(results, indent=2, ensure_ascii=False)
    

    def _get_document(self, docid: str):
        result = self.searcher.get_document(docid)
        if result is None:
            return json.dumps({"error": f"Document with docid '{docid}' not found"})
        return json.dumps(result, indent=2,ensure_ascii=False)

class RestaurantToolHandler:
    """
    A handler for encapsulating restaurant-related tool definitions and logic.
    """
    def __init__(self, restaurant_manager: Any):
        self.restaurant_manager = restaurant_manager

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        tools = [
                    {"type": "function",
                        "function": {
                            "name": "search_restaurants",
                            "description": "Get a list of recommended restaurants in a city based on its name.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {
                                        "type": "string",
                                        "description": "The city name, in Pinyin or English, e.g., 'Beijing'."
                                    }
                                },
                                "required": ["city"]
                            }
                        }
                    },
                    # Tool 2: Get restaurant details by name
                    {
                        "type": "function",
                        "function": {
                            "name": "get_restaurant_by_name",
                            "description": "Get detailed restaurant information based on city and restaurant name.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {
                                        "type": "string",
                                        "description": "The city name, in Pinyin or English, e.g., 'Beijing'."
                                    },
                                    "name": {
                                        "type": "string",
                                        "description": "English name of restaurant。"
                                    }
                                },
                                "required": ["city", "name"]
                            }
                        }
                    },
                    # Tool 3: Find restaurants by recommended food
                    {
                        "type": "function",
                        "function": {
                            "name": "find_restaurants_by_food",
                            "description": "Find restaurants that offer a particular dish, based on the city and the specific cuisine.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {
                                        "type": "string",
                                        "description": "The city name, in Pinyin or English, e.g., 'Beijing'."
                                    },
                                    "food": {
                                        "type": "string",
                                        "description": "food name"
                                    }
                                },
                                "required": ["city", "food"]
                            }
                        }
                    }
            ]
        return tools
    
    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Executes the corresponding restaurant query function based on the tool name and arguments"""
        if tool_name == "search_restaurants":
            result = self.restaurant_manager.list_all_restaurants_in_city(city=arguments["city"])
        elif tool_name == "get_restaurant_by_name":
            result = self.restaurant_manager.get_restaurant_details_by_name(city=arguments["city"], name=arguments["name"])

        elif tool_name == "find_restaurants_by_food":
            result = self.restaurant_manager.find_restaurants_by_recommended_food(city=arguments["city"], food=arguments["food"])
        else:
            raise ValueError(f"Unknown tool in RestaurantToolHandler : {tool_name}")
            

        return json.dumps(result, ensure_ascii=False, indent=2)

class ImageToolHandler:
    """
    A handler for encapsulating image retrieval and visual search tool definitions and logic.
    It interacts with the ImageIndexingManager to perform vector-based similarity searches.
    """
    def __init__(self, image_manager: Any):
        """
        Initializes the handler with an image manager instance.
        
        Args:
            image_manager: An instance of ImageIndexingManager that handles FAISS and DINOv2.
        """
        self.image_manager = image_manager

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Defines the tools available for the LLM to perform visual searches.
        """
        tools = [
            # Tool : Search by Image Path
            {
                "type": "function",
                "function": {
                    "name": "search_similar_images_by_path",
                    "description": "Find visually similar images by providing a local file path to a query image.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_path": {
                                "type": "string",
                                "description": "The absolute or relative local path to the image file."
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "The number of top similar images to retrieve. Defaults to 3.",
                                "default": 3
                            }
                        },
                        "required": ["image_path"]
                    }
                }
            }
        ]
        return tools

    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Executes the corresponding image search function based on the tool name.
        Includes safety type casting for LLM-provided arguments.
        """
        try:
            if tool_name == "search_similar_images_by_path":
                image_path = arguments["image_path"]
                top_k = int(arguments.get("top_k", 3))
                
                result = self.image_manager.search_similar_by_path(
                    query_img_path=image_path, 
                    k=top_k
                )
            else:
                raise ValueError(f"Unknown tool in ImageToolHandler: {tool_name}")

        except Exception as e:
            result = {"error": f"Failed to execute image tool: {str(e)}"}

        return json.dumps(result, ensure_ascii=False, indent=2)
    


class CombinedToolHandler:
    """
    A combined tool handler for managing and dispatching tasks to multiple independent tool handlers.
    """
    def __init__(self, handlers: list):
        """
        Initializes the combined handler.

        Args:
            handlers (list): A list containing all independent tool handler instances.
                             E.g., [SearchToolHandler(...), RestaurantToolHandler(...)]
        """
        self.handlers = handlers
        self._tool_to_handler_map = {}
        self._build_tool_map()

    def _build_tool_map(self):
        """Private method to build a map from tool names to their corresponding handlers for quick lookups."""
        for handler in self.handlers:
            # Assume each handler has a get_tool_definitions method.
            definitions = handler.get_tool_definitions()
            for tool_def in definitions:
                tool_name = tool_def['function']['name']
                if tool_name in self._tool_to_handler_map:
                    print(f"Warning: Tool name '{tool_name}' is duplicated. It will be overwritten by the latter handler.")
                self._tool_to_handler_map[tool_name] = handler

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Collects and returns a list of all tool definitions from all managed handlers."""
        all_tools = []
        for handler in self.handlers:
            all_tools.extend(handler.get_tool_definitions())
        return all_tools

    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Finds the handler responsible for the specified tool and delegates the tool call execution to it."""
        if tool_name not in self._tool_to_handler_map:
            raise ValueError(f"Tool not found in CombinedToolHandler: {tool_name}")
        
        # Find the correct handler
        target_handler = self._tool_to_handler_map[tool_name]
        
        # Delegate execution
        return target_handler.execute_tool(tool_name, arguments)
