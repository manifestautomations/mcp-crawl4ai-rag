-- Ensure the pgvector extension is enabled
create extension if not exists vector with schema extensions; -- Best practice to specify schema

-- Drop dependent functions first if they exist, to avoid errors during table recreation/alteration
drop function if exists public.match_crawled_pages(query_embedding vector, match_count integer, filter jsonb);
drop function if exists public.get_distinct_source_domains();

-- Recreate the documentation chunks table with the new embedding dimension
-- If the table already exists and you want to modify it, you might need to:
-- 1. Drop the old table (if data can be lost or is backed up)
--    DROP TABLE IF EXISTS public.crawled_pages;
-- 2. Or, ALTER the existing table (more complex if data types change significantly)
--    For a vector dimension change, dropping and recreating is often simpler if feasible.
--    If altering, you'd need to drop dependent indexes/constraints first.
-- For this script, we'll assume we can recreate it or it's a new setup.

create table if not exists public.crawled_pages (
    id bigserial primary key,
    url text not null, -- Using TEXT for potentially long URLs
    chunk_number integer not null,
    content text not null,  -- Stores the (potentially contextualized) chunk text that was embedded
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(384),  -- Adjusted for all-MiniLM-L6-v2 (384 dimensions)
                            -- Was vector(1536) for OpenAI text-embedding-3-small
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Unique constraint to prevent duplicate chunks for the same URL
    constraint unique_url_chunk unique(url, chunk_number)
);

-- Grant usage on the schema to the authenticated role if not already done
-- This might be necessary depending on your Supabase setup and roles
-- grant usage on schema public to authenticated;
-- grant select on table public.crawled_pages to authenticated;
-- grant all on table public.crawled_pages to service_role; -- Or your specific service role

-- Create indexes for performance
-- Index for vector similarity search (IVFFlat)
-- Note: The choice of lists for ivfflat depends on your data size.
-- For larger datasets, consider lists = sqrt(N) where N is number of rows, up to N/1000.
-- For smaller datasets, a smaller number of lists is fine.
-- Example: For up to 1M rows, lists = 100 might be okay. For 100M, lists = 1000.
-- probes determines how many lists are searched. Higher probes = better recall, slower query.
-- This index needs to be recreated if the table is dropped.
create index if not exists idx_crawled_pages_embedding_ivfflat on public.crawled_pages 
    using ivfflat (embedding public.vector_cosine_ops) -- Ensure vector_cosine_ops is in public schema or accessible
    with (lists = 100); -- Adjust 'lists' based on expected data size

-- Index for metadata filtering (GIN index)
create index if not exists idx_crawled_pages_metadata_gin on public.crawled_pages using gin (metadata);

-- Index specifically for filtering by source_domain within metadata, if frequently used
-- This assumes metadata structure like: {"source_domain": "example.com", ...}
create index if not exists idx_crawled_pages_metadata_source_domain on public.crawled_pages ((metadata->>'source_domain'));


-- Function to search for documentation chunks
-- This function now expects a 384-dimension query_embedding
create or replace function public.match_crawled_pages (
  query_embedding vector(384), -- Adjusted dimension
  match_count int default 10,
  filter jsonb default '{}'::jsonb -- Ensure filter is always a valid JSONB object
) returns table (
  id bigint,
  url text,
  chunk_number integer,
  content text,
  metadata jsonb,
  similarity float -- Cosine similarity
)
language plpgsql
as $$
#variable_conflict use_column 
begin
  -- Ensure the filter is not null and is a valid JSONB object
  -- An empty JSONB object '{}' means no filtering on metadata.
  -- If filter is NULL, default to '{}' to prevent errors with @>
  if filter is null then
    filter := '{}'::jsonb;
  end if;

  return query
  select
    cp.id,
    cp.url,
    cp.chunk_number,
    cp.content,
    cp.metadata,
    1 - (cp.embedding <=> query_embedding) as similarity -- Cosine distance is <->, so 1 - dist = similarity
  from public.crawled_pages cp -- Alias table for clarity
  where cp.metadata @> filter -- Only include if metadata contains all key/value pairs in filter
  order by cp.embedding <=> query_embedding -- Order by distance (ascending)
  limit match_count;
end;
$$;

-- Function to get distinct source domains
create or replace function public.get_distinct_source_domains()
returns table (source_domain text)
language sql
stable -- Indicates the function cannot modify the database and always returns the same results for the same arguments within a single scan
as $$
  select distinct metadata->>'source_domain' as source_domain
  from public.crawled_pages
  where metadata->>'source_domain' is not null;
$$;


-- Row Level Security (RLS) - Adjust as per your security model
-- Disable RLS first if you are making changes to policies or the table structure extensively
-- alter table public.crawled_pages disable row level security;

alter table public.crawled_pages enable row level security;

-- Example RLS Policies (REVIEW AND ADJUST THESE CAREFULLY)

-- Policy: Allow public read access (if your data is meant to be public)
-- Drop policy if it exists to recreate/update it
drop policy if exists "Allow public read access" on public.crawled_pages;
create policy "Allow public read access"
  on public.crawled_pages
  for select
  to public -- 'public' role means any user, including anonymous if anon key is used
  using (true); -- No specific condition, allows all rows to be selected

-- Policy: Allow service_role all access (typical for backend operations)
-- The service_role bypasses RLS by default, but explicit policies can be good for clarity or if RLS is forced.
-- Check your Supabase project settings. If service_role bypasses RLS, this might not be strictly needed.
drop policy if exists "Allow service_role full access" on public.crawled_pages;
create policy "Allow service_role full access"
  on public.crawled_pages
  for all -- Covers SELECT, INSERT, UPDATE, DELETE
  to service_role -- Or your specific backend role
  using (true)
  with check (true);

-- Grant permissions on functions to roles that need to call them
-- grant execute on function public.match_crawled_pages(vector, int, jsonb) to authenticated; -- If authenticated users need to search
grant execute on function public.match_crawled_pages(vector(384), int, jsonb) to service_role; -- Backend role

-- grant execute on function public.get_distinct_source_domains() to authenticated;
grant execute on function public.get_distinct_source_domains() to service_role;


-- After making schema changes, especially to vector dimensions or indexing strategy,
-- it's a good idea to re-analyze the table for optimal query planning.
-- ANALYZE public.crawled_pages;

comment on column public.crawled_pages.embedding is 'Vector embedding for the content chunk, 384 dimensions for all-MiniLM-L6-v2.';
comment on function public.match_crawled_pages(query_embedding vector(384), match_count integer, filter jsonb) is 'Searches for document chunks based on vector similarity (cosine similarity) and metadata filters. Expects a 384-dimensional query embedding.';

