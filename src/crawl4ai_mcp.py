"""
MCP server for web crawling with Crawl4AI.

This server provides tools to crawl websites using Crawl4AI, automatically detecting
the appropriate crawl method based on URL type (sitemap, txt file, or regular webpage).
Content is processed for RAG using local embeddings and OpenRouter for contextual summaries.
"""
from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from dotenv import load_dotenv
from supabase import Client
from pathlib import Path
import requests
import asyncio
import json
import os
import re
import datetime # For UTC timestamps
import traceback # For detailed error logging

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, RateLimiter
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher # Using this dispatcher

# Import utility functions (now updated for OpenRouter and local embeddings)
from utils import get_supabase_client, add_documents_to_supabase, search_documents, EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

# --- Application Context ---
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler
    supabase_client: Client

# --- Lifespan Management for MCP Server ---
@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        Crawl4AIContext: The context containing the Crawl4AI crawler and Supabase client
    """
    # Create browser configuration
    browser_config = BrowserConfig(
        headless=True,
        verbose=False # Set to True for more detailed browser logs from Crawl4AI
    )
    
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__() # Initialize the crawler
    
    supabase_client = get_supabase_client()
    
    try:
        # Use timezone-aware UTC time
        print(f"Crawl4AI MCP Server lifespan started at {datetime.datetime.now(datetime.UTC)} UTC.")
        print(f"Using embedding model: {EMBEDDING_MODEL_NAME} (Dimension: {EMBEDDING_DIMENSION})")
        if os.getenv("OPENROUTER_API_KEY") and os.getenv("LLM_MODEL_CHOICE"):
            print(f"Contextual info generation enabled with OpenRouter model: {os.getenv('LLM_MODEL_CHOICE')}")
        else:
            print("Contextual info generation via LLM is disabled (OPENROUTER_API_KEY or LLM_MODEL_CHOICE not set).")
        
        yield Crawl4AIContext(
            crawler=crawler,
            supabase_client=supabase_client
        )
    finally:
        await crawler.__aexit__(None, None, None) # Properly close the crawler
        # Use timezone-aware UTC time
        print(f"Crawl4AI MCP Server lifespan ended at {datetime.datetime.now(datetime.UTC)} UTC.")

# --- MCP Server Initialization ---
mcp = FastMCP(
    "mcp-crawl4ai-rag",
    description="MCP server for RAG and web crawling with Crawl4AI, OpenRouter, and local embeddings.",
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=int(os.getenv("PORT", "8051")) # Ensure port is an integer
)

# --- Helper Functions for Crawling ---
def is_sitemap(url: str) -> bool:
    """Check if a URL points to a sitemap.xml file."""
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path.lower()

def is_txt(url: str) -> bool:
    """Check if a URL points to a .txt file."""
    return url.endswith('.txt')

def parse_sitemap(sitemap_url: str) -> List[str]:
    """Parse a sitemap.xml and extract all URLs."""
    urls = []
    try:
        headers = {'User-Agent': 'Crawl4AI-MCP-Agent/1.0'} 
        resp = requests.get(sitemap_url, headers=headers, timeout=15) 
        resp.raise_for_status() 
        
        tree = ElementTree.fromstring(resp.content)
        urls = [loc.text for loc in tree.findall('.//{*}loc') if loc.text]
        print(f"Parsed {len(urls)} URLs from sitemap: {sitemap_url}")
    except requests.RequestException as e:
        print(f"Error fetching sitemap {sitemap_url}: {e}")
    except ElementTree.ParseError as e:
        print(f"Error parsing XML from sitemap {sitemap_url}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while parsing sitemap {sitemap_url}: {e}")
    return urls

def smart_chunk_markdown(text: str, chunk_size_chars: int = 5000, overlap_chars: int = 200) -> List[str]:
    """
    Split text into chunks, respecting Markdown structure (code blocks, paragraphs)
    and adding overlap.
    
    Args:
        text: The Markdown text to chunk.
        chunk_size_chars: Target size for each chunk in characters.
        overlap_chars: Number of characters to overlap between chunks.
        
    Returns:
        A list of text chunks.
    """
    if not text:
        return []

    chunks = []
    current_pos = 0
    text_len = len(text)

    while current_pos < text_len:
        end_pos = min(current_pos + chunk_size_chars, text_len)
        
        if end_pos == text_len: 
            chunk = text[current_pos:]
            if chunk.strip(): 
                 chunks.append(chunk.strip())
            break 

        split_pos = -1
        
        code_block_search_limit = max(current_pos, end_pos - chunk_size_chars + overlap_chars) 
        last_code_block_end = text.rfind('```\n', code_block_search_limit, end_pos)
        if last_code_block_end != -1:
            split_pos = last_code_block_end + 4 
        
        if split_pos == -1:
            paragraph_search_limit = max(current_pos, end_pos - chunk_size_chars + overlap_chars)
            last_paragraph_break = text.rfind('\n\n', paragraph_search_limit, end_pos)
            if last_paragraph_break != -1:
                split_pos = last_paragraph_break + 2 

        if split_pos == -1:
            sentence_search_limit = max(current_pos, end_pos - chunk_size_chars + overlap_chars)
            sentence_breaks = [
                m.end() for m in re.finditer(r'[.!?](?=\s|\n|$)', text[sentence_search_limit:end_pos])
            ]
            if sentence_breaks:
                split_pos = sentence_search_limit + sentence_breaks[-1]
        
        if split_pos == -1 or split_pos <= current_pos: 
            last_space_search_limit = max(current_pos, end_pos - chunk_size_chars + overlap_chars)
            last_space = text.rfind(' ', last_space_search_limit, end_pos)
            if last_space != -1:
                split_pos = last_space + 1 
            else: 
                split_pos = end_pos
        
        if split_pos <= current_pos :
            split_pos = end_pos 

        chunk = text[current_pos:split_pos]
        if chunk.strip(): 
            chunks.append(chunk.strip())
        
        next_start_pos = split_pos - overlap_chars
        if next_start_pos <= current_pos and split_pos < text_len: 
             current_pos = split_pos 
        else:
             current_pos = next_start_pos
        
        if current_pos >= text_len: 
            break
            
    return [c for c in chunks if c] 

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """Extracts headers and basic stats from a Markdown chunk."""
    headers = re.findall(r'^(#{1,6})\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1].strip()}' for h in headers]) if headers else 'N/A'

    return {
        "headers_in_chunk": header_str,
        "char_count_chunk": len(chunk),
        "word_count_chunk": len(chunk.split())
    }

# --- MCP Tools ---
@mcp.tool()
async def crawl_single_page(ctx: Context, url: str, chunk_size_chars: int = 4000, chunk_overlap_chars: int = 200) -> str:
    """
    Crawl a single web page, chunk its content, and store it in Supabase with local embeddings.
    """
    try:
        crawler_context = ctx.request_context.lifespan_context
        crawler = crawler_context.crawler
        supabase_client = crawler_context.supabase_client
        
        print(f"Crawling single page: {url}")
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        
        result = await crawler.arun(url=url, config=run_config)
        
        if result.success and result.markdown:
            print(f"Successfully crawled {url}. Content length: {len(result.markdown)} chars.")
            chunks = smart_chunk_markdown(result.markdown, chunk_size_chars=chunk_size_chars, overlap_chars=chunk_overlap_chars)
            print(f"Split content into {len(chunks)} chunks.")
            
            urls_for_db = []
            chunk_numbers_for_db = []
            contents_for_db = [] 
            metadatas_for_db = []
            
            for i, chunk_content in enumerate(chunks):
                urls_for_db.append(url)
                chunk_numbers_for_db.append(i)
                contents_for_db.append(chunk_content)
                
                meta = extract_section_info(chunk_content)
                meta["chunk_index_in_doc"] = i 
                meta["original_url"] = url 
                meta["source_domain"] = urlparse(url).netloc 
                meta["crawl_tool"] = "crawl_single_page" 
                meta["crawled_at_utc"] = datetime.datetime.now(datetime.UTC).isoformat() # Use timezone-aware UTC
                metadatas_for_db.append(meta)
            
            url_to_full_document = {url: result.markdown}
            
            add_documents_to_supabase(
                supabase_client, 
                urls_for_db, 
                chunk_numbers_for_db, 
                contents_for_db, 
                metadatas_for_db, 
                url_to_full_document,
                batch_size=20 
            )
            
            return json.dumps({
                "success": True,
                "url": url,
                "chunks_processed_for_storage": len(chunks),
                "original_content_length_chars": len(result.markdown),
                "internal_links_found": len(result.links.get("internal", [])),
                "external_links_found": len(result.links.get("external", []))
            }, indent=2)
        else:
            error_msg = result.error_message if result and hasattr(result, 'error_message') else "Unknown error during crawl."
            print(f"Failed to crawl {url}: {error_msg}")
            return json.dumps({"success": False, "url": url, "error": error_msg}, indent=2)
    except Exception as e:
        print(f"Exception in crawl_single_page for {url}: {e}")
        return json.dumps({"success": False, "url": url, "error": str(e), "trace": traceback.format_exc()}, indent=2)

@mcp.tool()
async def smart_crawl_url(
    ctx: Context, 
    url: str, 
    max_depth_recursive: int = 2, 
    max_concurrent_sessions: int = 5, 
    chunk_size_chars: int = 4000,
    chunk_overlap_chars: int = 200,
    page_limit_sitemap: int = 1000, 
    page_limit_recursive: int = 100  
) -> str:
    """
    Intelligently crawl a URL (webpage, sitemap, or .txt file), chunk content, and store in Supabase.
    """
    try:
        crawler_context = ctx.request_context.lifespan_context
        crawler = crawler_context.crawler
        supabase_client = crawler_context.supabase_client
        
        print(f"Starting smart_crawl_url for: {url} with max_concurrent_sessions: {max_concurrent_sessions}")
        
        crawl_results_docs = [] 
        crawl_type_detected = "unknown"

        rate_limiter_config = RateLimiter(
            base_delay=(0.5, 1.5), 
            max_delay=30.0,      
            max_retries=3
        )
        
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=80.0, 
            check_interval=2.0,          
            max_session_permit=max_concurrent_sessions,
            rate_limiter=rate_limiter_config 
        )

        if is_txt(url):
            crawl_type_detected = "text_file"
            print(f"Detected text file: {url}. Fetching directly.")
            run_config_txt = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False) 
            result = await crawler.arun(url=url, config=run_config_txt)
            if result.success and result.markdown:
                crawl_results_docs.append({'url': url, 'markdown': result.markdown})
            else:
                error_msg = result.error_message if result and hasattr(result, 'error_message') else "Unknown error fetching text file."
                print(f"Failed to fetch text file {url}: {error_msg}")

        elif is_sitemap(url):
            crawl_type_detected = "sitemap"
            print(f"Detected sitemap: {url}. Parsing and crawling URLs.")
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps({"success": False, "url": url, "error": "No URLs found in sitemap or sitemap parsing failed."}, indent=2)
            
            print(f"Found {len(sitemap_urls)} URLs in sitemap. Limiting to {page_limit_sitemap} URLs for this crawl.")
            urls_to_crawl_sitemap = sitemap_urls[:page_limit_sitemap]
            
            if urls_to_crawl_sitemap:
                run_config_sitemap = CrawlerRunConfig(cache_mode=CacheMode.ENABLED, stream=False) 
                batch_results = await crawler.arun_many(urls=urls_to_crawl_sitemap, config=run_config_sitemap, dispatcher=dispatcher)
                crawl_results_docs.extend([{'url': r.url, 'markdown': r.markdown} for r in batch_results if r.success and r.markdown])
            else:
                 print("No URLs to crawl from sitemap after filtering/limiting.")

        else: 
            crawl_type_detected = "webpage_recursive"
            print(f"Detected webpage: {url}. Starting recursive crawl (max_depth: {max_depth_recursive}, page_limit: {page_limit_recursive}).")
            run_config_recursive = CrawlerRunConfig(cache_mode=CacheMode.ENABLED, stream=False) 
            
            visited_urls = set()
            start_url_normalized = urldefrag(url)[0]
            urls_to_process_queue = [start_url_normalized] 
            all_pages_markdown = {} 

            for depth in range(max_depth_recursive + 1): 
                if not urls_to_process_queue or len(all_pages_markdown) >= page_limit_recursive:
                    print(f"Exiting recursive crawl at depth {depth}. Queue empty or page limit ({page_limit_recursive}) reached.")
                    break
                
                current_level_urls_to_crawl = list(set(urls_to_process_queue)) 
                urls_to_process_queue = [] 
                
                # Corrected logic for creating crawl_batch_urls
                # 1. Filter candidates not yet visited
                candidate_urls_for_batch = [u for u in current_level_urls_to_crawl if u not in visited_urls]
                
                # 2. Determine how many more URLs we can take based on page_limit_recursive
                remaining_capacity = page_limit_recursive - len(all_pages_markdown)
                
                # 3. Form the actual batch, taking only up to the remaining capacity
                crawl_batch_urls = candidate_urls_for_batch[:max(0, remaining_capacity)]

                if not crawl_batch_urls:
                    print(f"No new URLs to crawl at depth {depth} or page limit reached for this level.")
                    continue # Skip to next depth or finish if no more URLs overall

                print(f"Recursive crawl depth {depth}: Attempting to crawl {len(crawl_batch_urls)} URLs.")
                batch_crawl_results = await crawler.arun_many(urls=crawl_batch_urls, config=run_config_recursive, dispatcher=dispatcher)
                
                new_links_found_for_next_level = set()

                for res in batch_crawl_results:
                    norm_res_url = urldefrag(res.url)[0]
                    visited_urls.add(norm_res_url) 

                    if res.success and res.markdown:
                        if norm_res_url not in all_pages_markdown and len(all_pages_markdown) < page_limit_recursive:
                            all_pages_markdown[norm_res_url] = res.markdown
                            print(f"  Successfully crawled and stored: {norm_res_url} (Total stored: {len(all_pages_markdown)})")
                            if depth < max_depth_recursive:
                                for link_info in res.links.get("internal", []):
                                    abs_link = link_info.get("href")
                                    if abs_link:
                                        norm_link = urldefrag(abs_link)[0]
                                        if urlparse(norm_link).netloc == urlparse(start_url_normalized).netloc and \
                                           norm_link not in visited_urls and \
                                           norm_link not in current_level_urls_to_crawl and \
                                           norm_link not in new_links_found_for_next_level: # Check against links already added for next level
                                            new_links_found_for_next_level.add(norm_link)
                        elif norm_res_url in all_pages_markdown:
                            print(f"  Skipped storing (already processed): {norm_res_url}")
                        else: 
                            print(f"  Crawled {norm_res_url}, but page limit reached. Not storing.")
                    else:
                        print(f"  Failed to crawl: {norm_res_url} (Error: {res.error_message if hasattr(res, 'error_message') else 'Unknown'})")
                
                urls_to_process_queue.extend(list(new_links_found_for_next_level))


            crawl_results_docs = [{'url': u, 'markdown': md} for u, md in all_pages_markdown.items()]

        if not crawl_results_docs:
            return json.dumps({"success": False, "url": url, "error": "No content successfully crawled or extracted."}, indent=2)
        
        print(f"Successfully crawled {len(crawl_results_docs)} pages. Processing for Supabase.")
        
        all_urls_for_db = []
        all_chunk_numbers_for_db = []
        all_original_contents_for_db = [] 
        all_metadatas_for_db = []
        total_chunks_for_storage = 0
        
        url_to_full_document_map = {doc['url']: doc['markdown'] for doc in crawl_results_docs}

        for doc_data in crawl_results_docs:
            source_doc_url = doc_data['url']
            markdown_content = doc_data['markdown']
            
            chunks = smart_chunk_markdown(markdown_content, chunk_size_chars=chunk_size_chars, overlap_chars=chunk_overlap_chars)
            
            for i, chunk_content in enumerate(chunks):
                all_urls_for_db.append(source_doc_url)
                all_chunk_numbers_for_db.append(i)
                all_original_contents_for_db.append(chunk_content)
                
                meta = extract_section_info(chunk_content)
                meta["chunk_index_in_doc"] = i
                meta["original_url"] = source_doc_url
                meta["source_domain"] = urlparse(source_doc_url).netloc
                meta["crawl_tool"] = "smart_crawl_url"
                meta["detected_crawl_type"] = crawl_type_detected
                meta["crawled_at_utc"] = datetime.datetime.now(datetime.UTC).isoformat() # Use timezone-aware UTC
                all_metadatas_for_db.append(meta)
                total_chunks_for_storage += 1
        
        add_documents_to_supabase(
            supabase_client, 
            all_urls_for_db, 
            all_chunk_numbers_for_db, 
            all_original_contents_for_db, 
            all_metadatas_for_db, 
            url_to_full_document_map,
            batch_size=20 
        )
        
        return json.dumps({
            "success": True,
            "initial_url_crawled": url,
            "detected_crawl_type": crawl_type_detected,
            "pages_successfully_crawled_and_processed": len(crawl_results_docs),
            "total_chunks_processed_for_storage": total_chunks_for_storage,
            "crawled_urls_sample": [doc['url'] for doc in crawl_results_docs][:5] + (["..."] if len(crawl_results_docs) > 5 else [])
        }, indent=2)
        
    except Exception as e:
        print(f"Exception in smart_crawl_url for {url}: {e}")
        print(traceback.format_exc()) 
        return json.dumps({"success": False, "url": url, "error": str(e), "trace": traceback.format_exc()}, indent=2)


@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """
    Get all unique source domains from the crawled content in Supabase.
    """
    try:
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        print("Fetching available sources from Supabase...")
        
        result = supabase_client.rpc('get_distinct_source_domains').execute()

        if hasattr(result, 'data') and result.data is not None: 
            sources = [item['source_domain'] for item in result.data if item.get('source_domain')]
            sources = sorted(list(set(sources))) 
            print(f"Found {len(sources)} unique sources.")
            return json.dumps({"success": True, "sources": sources, "count": len(sources)}, indent=2)
        elif hasattr(result, 'error') and result.error:
            print(f"Error calling get_distinct_source_domains RPC: {result.error}")
            return json.dumps({"success": False, "error": f"RPC Error: {result.error.message}", "details": str(result.error)}, indent=2)
        else: 
            print("No distinct sources found or unexpected result from RPC call (result.data might be None or empty).")
            return json.dumps({"success": True, "sources": [], "count": 0, "message": "No sources found or data was empty/None."}, indent=2)
            
    except Exception as e:
        print(f"Error in get_available_sources: {e}")
        return json.dumps({"success": False, "error": str(e), "trace": traceback.format_exc()}, indent=2)

@mcp.tool()
async def perform_rag_query(ctx: Context, query: str, source_filter: Optional[str] = None, match_count: int = 5) -> str:
    """
    Perform a RAG query on stored content using local embeddings. Optionally filter by source domain.
    """
    try:
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        print(f"Performing RAG query: '{query[:50]}...', source_filter: {source_filter}, match_count: {match_count}")
        
        filter_metadata_for_search = None
        if source_filter and source_filter.strip():
            filter_metadata_for_search = {"source_domain": source_filter.strip()}
        
        results = search_documents(
            client=supabase_client,
            query=query,
            match_count=match_count,
            filter_metadata=filter_metadata_for_search
        )
        
        formatted_results = []
        if results: 
            for res_item in results:
                if isinstance(res_item, dict):
                    formatted_results.append({
                        "url": res_item.get("url"),
                        "chunk_content": res_item.get("content"), 
                        "metadata": res_item.get("metadata"),
                        "similarity_score": res_item.get("similarity") 
                    })
                else:
                    print(f"Warning: Unexpected item format in search results: {res_item}")

        print(f"RAG query returned {len(formatted_results)} formatted results.")
        return json.dumps({
            "success": True,
            "query": query,
            "applied_source_filter": source_filter if source_filter and source_filter.strip() else "None",
            "results_count": len(formatted_results),
            "results": formatted_results
        }, indent=2)
    except Exception as e:
        print(f"Error in perform_rag_query: {e}")
        print(traceback.format_exc())
        return json.dumps({"success": False, "query": query, "error": str(e), "trace": traceback.format_exc()}, indent=2)

# --- Main Execution ---
async def main():
    """Main function to run the MCP server."""
    transport = os.getenv("TRANSPORT", "sse").lower()
    print(f"Starting MCP server with {transport} transport...")
    if transport == 'sse':
        await mcp.run_sse_async()
    elif transport == 'stdio':
        await mcp.run_stdio_async()
    else:
        print(f"Unsupported TRANSPORT: {transport}. Defaulting to sse.")
        await mcp.run_sse_async()

if __name__ == "__main__":
    asyncio.run(main())
