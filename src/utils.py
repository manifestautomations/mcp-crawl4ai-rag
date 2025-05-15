"""
Utility functions for the Crawl4AI MCP server.
"""
import os
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import json
from supabase import create_client, Client
from urllib.parse import urlparse
import openai # We'll still use the openai library, but reconfigured for OpenRouter
from sentence_transformers import SentenceTransformer # For local embeddings

# --- Configuration ---
# LLM Configuration (OpenRouter)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1" # OpenRouter API endpoint
LLM_MODEL_CHOICE = os.getenv("LLM_MODEL_CHOICE", "anthropic/claude-3.5-sonnet") # Default to a capable model on OpenRouter

# Embedding Configuration (Local)
# It's good practice to load the model once and reuse it.
# Consider initializing this in a more global context if utils.py is imported multiple times
# or if these functions are called very frequently in a way that re-initializes.
# For simplicity in this example, we'll load it as needed but with a global-like placeholder.
_embedding_model = None
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Produces 384-dimensional embeddings
EMBEDDING_DIMENSION = 384 # Dimension for all-MiniLM-L6-v2

def get_embedding_model():
    """Loads and returns the sentence-transformer model."""
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading local embedding model: {EMBEDDING_MODEL_NAME}...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded.")
    return _embedding_model

# --- Supabase Client ---
def get_supabase_client() -> Client:
    """
    Get a Supabase client with the URL and key from environment variables.
    
    Returns:
        Supabase client instance
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables")
    
    return create_client(url, key)

# --- OpenAI Client for OpenRouter ---
def get_openrouter_client() -> openai.OpenAI:
    """
    Get an OpenAI client configured for OpenRouter.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY must be set in environment variables for LLM calls.")
    
    return openai.OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )

# --- Embedding Functions (Local) ---
def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts in a single call using a local Sentence Transformer model.
    
    Args:
        texts: List of texts to create embeddings for
        
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []
        
    try:
        model = get_embedding_model()
        embeddings = model.encode(texts, show_progress_bar=False) # Set show_progress_bar to True for long batches
        return [emb.tolist() for emb in embeddings]
    except Exception as e:
        print(f"Error creating batch embeddings locally: {e}")
        # Return zero vectors if there's an error, matching the configured dimension
        return [[0.0] * EMBEDDING_DIMENSION for _ in range(len(texts))]

def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using a local Sentence Transformer model.
    
    Args:
        text: Text to create an embedding for
        
    Returns:
        List of floats representing the embedding
    """
    try:
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.0] * EMBEDDING_DIMENSION
    except Exception as e:
        print(f"Error creating local embedding: {e}")
        # Return zero vector if there's an error
        return [0.0] * EMBEDDING_DIMENSION

# --- Contextual Information Generation (OpenRouter LLM) ---
def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document using OpenRouter.
    This function name is a bit of a misnomer now as it generates *text* for embedding,
    not the embedding itself.
    
    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual information generation was performed
    """
    try:
        client = get_openrouter_client()
        # Create the prompt for generating contextual information
        # Truncate full_document to avoid excessive token usage/cost
        # Max context for many models is around 8k-32k tokens, but prompts should be shorter.
        # ~25000 characters is roughly 6k-8k tokens. Adjust as needed.
        max_doc_len_for_prompt = 25000
        prompt = f"""<document> 
{full_document[:max_doc_len_for_prompt]} 
</document>
Here is the chunk we want to situate within the whole document:
<chunk> 
{chunk}
</chunk> 
Please give a short, succinct context (1-2 sentences) to situate this chunk within the overall document. This context will be prepended to the chunk to improve search retrieval. Answer ONLY with the succinct context. Do not add any other explanatory text. Example: If the document is about Python programming and the chunk is about list comprehensions, a good context would be: "This section details list comprehensions, a concise way to create lists in Python." """

        # Call the OpenRouter API (via OpenAI client)
        response = client.chat.completions.create(
            model=LLM_MODEL_CHOICE,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise contextual information for text chunks. Your response should be only the contextual sentence(s)."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # Lower temperature for more factual, less creative context
            max_tokens=150  # Max tokens for the context itself
        )
        
        context = response.choices[0].message.content.strip()
        
        # Combine the context with the original chunk
        # Ensure there's a clear separator.
        contextual_text = f"Context: {context}\n---\nOriginal Chunk:\n{chunk}"
        
        print(f"Generated context for chunk: {context[:100]}...") # Log a snippet
        return contextual_text, True
    
    except Exception as e:
        print(f"Error generating contextual information via OpenRouter: {e}. Using original chunk instead.")
        return chunk, False

def process_chunk_with_context(args):
    """
    Process a single chunk, potentially adding contextual information.
    This function is designed to be used with concurrent.futures.
    
    Args:
        args: Tuple containing (url, content, full_document)
        
    Returns:
        Tuple containing:
        - The (potentially contextualized) text of the chunk
        - Boolean indicating if contextual information generation was performed
    """
    url, content, full_document = args # Unpack arguments
    # Only generate context if LLM_MODEL_CHOICE is set, implying user wants this feature
    if LLM_MODEL_CHOICE and OPENROUTER_API_KEY:
        return generate_contextual_embedding(full_document, content)
    else:
        # If no LLM is configured for context, return original chunk
        return content, False


# --- Supabase Data Handling ---
def add_documents_to_supabase(
    client: Client, 
    urls: List[str], 
    chunk_numbers: List[int],
    contents: List[str], 
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 20 # Batch for Supabase insertion
) -> None:
    """
    Add documents to the Supabase crawled_pages table in batches.
    Deletes existing records with the same URLs before inserting to prevent duplicates.
    Generates contextual information if LLM is configured, then creates local embeddings.
    
    Args:
        client: Supabase client
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of original document contents (chunks)
        metadatas: List of document metadata for each chunk
        url_to_full_document: Dictionary mapping URLs to their full document content
        batch_size: Size of each batch for Supabase insertion
    """
    # Get unique URLs to delete existing records
    unique_urls = list(set(urls)) # Remove duplicates for deletion query
    
    # Delete existing records for these URLs in a single operation if possible
    try:
        if unique_urls:
            print(f"Deleting existing records for {len(unique_urls)} URLs...")
            # Use the .in_() filter to delete all records with matching URLs
            client.table("crawled_pages").delete().in_("url", unique_urls).execute()
            print("Deletion complete.")
    except Exception as e:
        print(f"Batch delete failed: {e}. Trying one-by-one deletion as fallback.")
        for url_to_delete in unique_urls:
            try:
                client.table("crawled_pages").delete().eq("url", url_to_delete).execute()
            except Exception as inner_e:
                print(f"Error deleting record for URL {url_to_delete}: {inner_e}")
    
    # Determine if contextual information generation is enabled
    use_contextual_info = bool(LLM_MODEL_CHOICE and OPENROUTER_API_KEY)
    
    # Process in batches to avoid memory issues and manage API calls
    for i in range(0, len(contents), batch_size):
        batch_start_index = i
        batch_end_index = min(i + batch_size, len(contents))
        
        current_batch_urls = urls[batch_start_index:batch_end_index]
        current_batch_chunk_numbers = chunk_numbers[batch_start_index:batch_end_index]
        current_batch_original_contents = contents[batch_start_index:batch_end_index]
        current_batch_metadatas = metadatas[batch_start_index:batch_end_index]
        
        texts_to_embed = [] # This list will hold the content that gets embedded

        if use_contextual_info:
            print(f"Generating contextual info for batch {i//batch_size + 1}...")
            # Prepare arguments for parallel processing of contextual info generation
            process_args = []
            for j, original_chunk_content in enumerate(current_batch_original_contents):
                url_of_chunk = current_batch_urls[j]
                full_doc_content = url_to_full_document.get(url_of_chunk, "")
                process_args.append((url_of_chunk, original_chunk_content, full_doc_content))
            
            # Process in parallel using ThreadPoolExecutor for contextual info
            # Max workers for LLM calls should be modest to avoid rate limits on OpenRouter
            # and respect its terms of service.
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 5) as executor:
                future_to_idx = {
                    executor.submit(process_chunk_with_context, arg): original_idx 
                    for original_idx, arg in enumerate(process_args)
                }
                
                # Initialize results list with original content as fallback
                batch_processed_texts = list(current_batch_original_contents) 
                
                for future in concurrent.futures.as_completed(future_to_idx):
                    original_idx = future_to_idx[future]
                    try:
                        processed_text, context_generated = future.result()
                        batch_processed_texts[original_idx] = processed_text # Update with processed text
                        current_batch_metadatas[original_idx]["contextual_info_generated"] = context_generated
                    except Exception as e:
                        print(f"Error processing chunk for context (index {original_idx}): {e}")
                        # Fallback to original content is already in batch_processed_texts[original_idx]
                        current_batch_metadatas[original_idx]["contextual_info_generated"] = False
            texts_to_embed = batch_processed_texts
        else:
            # If not using contextual info, embed the original contents
            texts_to_embed = current_batch_original_contents
            for k in range(len(current_batch_metadatas)):
                 current_batch_metadatas[k]["contextual_info_generated"] = False
        
        # Create embeddings for the (potentially contextualized) batch using local model
        print(f"Creating local embeddings for batch {i//batch_size + 1} ({len(texts_to_embed)} items)...")
        batch_embeddings = create_embeddings_batch(texts_to_embed)
        print("Embeddings created.")
        
        # Prepare data for Supabase insertion
        supabase_batch_data = []
        for j in range(len(texts_to_embed)):
            # The content stored in Supabase should be the one that was embedded.
            # Metadata should reflect this.
            chunk_text_for_storage = texts_to_embed[j]
            chunk_size = len(chunk_text_for_storage)
            
            # Ensure metadata is a dictionary
            meta_entry = current_batch_metadatas[j] if isinstance(current_batch_metadatas[j], dict) else {}

            data_for_supabase = {
                "url": current_batch_urls[j],
                "chunk_number": current_batch_chunk_numbers[j],
                "content": chunk_text_for_storage, # Store the text that was embedded
                "metadata": {
                    "chunk_size_chars": chunk_size, # Renamed for clarity
                    **meta_entry # Spread the existing metadata
                },
                "embedding": batch_embeddings[j] 
            }
            supabase_batch_data.append(data_for_supabase)
        
        # Insert batch into Supabase
        try:
            if supabase_batch_data:
                print(f"Inserting batch {i//batch_size + 1} into Supabase ({len(supabase_batch_data)} records)...")
                client.table("crawled_pages").insert(supabase_batch_data).execute()
                print("Batch insertion successful.")
            else:
                print(f"Skipping insertion for batch {i//batch_size + 1} as it's empty.")
        except Exception as e:
            print(f"Error inserting batch into Supabase: {e}")
            # Optionally, add more robust error handling here, like retrying individual records

def search_documents(
    client: Client, 
    query: str, 
    match_count: int = 10, 
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for documents in Supabase using vector similarity with local embeddings.
    
    Args:
        client: Supabase client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter (e.g., {"source": "example.com"})
        
    Returns:
        List of matching documents
    """
    # Create embedding for the query using the local model
    print(f"Creating local embedding for query: {query[:50]}...")
    query_embedding = create_embedding(query)
    print("Query embedding created.")
    
    # Execute the search using the match_crawled_pages function
    try:
        params = {
            'query_embedding': query_embedding, # This is now a list of floats
            'match_count': match_count
        }
        
        # Only add the filter if it's actually provided and not empty
        if filter_metadata and isinstance(filter_metadata, dict) and filter_metadata:
            params['filter'] = filter_metadata
        else:
            params['filter'] = {} # Supabase function expects a filter, even if empty
        
        print(f"Searching Supabase with params: query_embedding (shape: {len(query_embedding)}), match_count: {match_count}, filter: {params.get('filter')}")
        result = client.rpc('match_crawled_pages', params).execute()
        
        if hasattr(result, 'data'):
            print(f"Search returned {len(result.data)} results.")
            return result.data
        else:
            print(f"Search result did not have 'data' attribute. Result: {result}")
            return []
            
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []

