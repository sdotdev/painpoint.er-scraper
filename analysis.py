# analysis.py
import json # Need this for merging JSON results

# Define the core analysis prompt (Remains the same)
ANALYSIS_SYSTEM_PROMPT = """
You are a product strategy consultant analyzing user comments.
Your goal is to identify potential market opportunities for new software products or features.
Analyze the following text scraped from online discussions (like Reddit).
Focus specifically on:
1.  Products or software people mention they currently use but are unhappy with or looking for alternatives to.
2.  Specific features people wish existing products had.
3.  Pain points mentioned regarding existing software or workflows.
4.  Recurring requests or suggestions for new types of tools or functionalities.

Summarize these findings clearly, highlighting actionable insights for software developers or product managers looking for proven product ideas to build and sell. Structure your output logically, perhaps grouping by product category or type of opportunity. Be concise and focus on the opportunities.
""" # Note: The Azure client adds JSON instructions separately

# --- Configuration for Chunking ---
# Estimate max characters per chunk to stay well under token limits (including prompts/schema)
# ~8000 tokens limit. Let's aim for ~15k chars for user content (~4k tokens), leaving ample room.
MAX_CHARS_PER_CHUNK = 15000
COMMENT_SEPARATOR = "\n---\n" # Separator used between comments within a chunk

def merge_analysis_results(results_list):
    """
    Merges analysis results from multiple chunks.
    Handles both string results (concatenates) and dictionary results (merges lists within).
    """
    if not results_list:
        return None

    # Check the type of the first valid result to determine merge strategy
    first_valid_result = next((res for res in results_list if res is not None), None)
    if first_valid_result is None:
        return None # No valid results to merge

    if isinstance(first_valid_result, str):
        # Concatenate string results
        return "\n\n==== Chunk Separator ====\n\n".join(
            res for res in results_list if isinstance(res, str) # Ensure only strings are joined
        )
    elif isinstance(first_valid_result, dict):
        # Merge dictionary results (assuming JSON schema structure)
        # Initialize with the structure of the schema
        combined_result = {
            "product_opportunities": [],
            "feature_requests": [],
            "pain_points": [],
            "new_tool_suggestions": [],
            "_errors": [] # Add a field to collect errors from chunks
        }
        schema_keys = set(combined_result.keys()) - {"_errors"}

        for i, result in enumerate(results_list):
            if isinstance(result, dict):
                # Check for errors reported by the client itself
                if "error" in result:
                    combined_result["_errors"].append({f"chunk_{i+1}": result})
                    continue

                # Merge lists for each key defined in our schema
                for key in schema_keys:
                    if key in result and isinstance(result[key], list):
                        combined_result[key].extend(result[key])
                    elif key in result:
                         # Log if the key exists but isn't a list as expected
                         print(f"Warning: Chunk {i+1} result for key '{key}' was not a list: {type(result[key])}")
                         combined_result["_errors"].append({f"chunk_{i+1}_warning": f"Key '{key}' not a list."})

            elif result is not None: # Handle case where a chunk returned non-dict unexpectedly
                print(f"Warning: Chunk {i+1} result was not a dictionary: {type(result)}")
                combined_result["_errors"].append({f"chunk_{i+1}_error": "Result was not a dictionary."})

        # Clean up empty error list if no errors occurred
        if not combined_result["_errors"]:
            del combined_result["_errors"]

        return combined_result
    else:
        # Fallback for unexpected type
        print(f"Warning: Cannot merge results of unknown type: {type(first_valid_result)}")
        return str(results_list)


def perform_analysis(ai_client, scraped_comments_list):
    """
    Prepares data, chunks it if necessary, sends chunks to the AI client
    for analysis, and merges the results.

    Args:
        ai_client (AIClient): An instance of an AI client (OpenAI, AzureGitHub, etc.).
        scraped_comments_list (list[str]): A list of scraped comment strings.

    Returns:
        str or dict or None: The combined analysis result, or None if no analysis could be performed.
    """
    if not scraped_comments_list:
        print("No comments provided for analysis.")
        return None

    all_analysis_results = []
    current_chunk_comments = []
    current_chunk_chars = 0
    separator_len = len(COMMENT_SEPARATOR)

    print(f"Processing {len(scraped_comments_list)} comments...")

    for i, comment in enumerate(scraped_comments_list):
        comment_len = len(comment)
        # Calculate potential size *if* this comment is added
        potential_new_chars = current_chunk_chars + (separator_len if current_chunk_comments else 0) + comment_len

        # --- Chunking Logic ---
        # If the chunk is not empty AND adding the current comment would exceed the limit...
        if current_chunk_comments and potential_new_chars > MAX_CHARS_PER_CHUNK:
            # 1. Process the current chunk
            print(f"Processing chunk {len(all_analysis_results) + 1} ({(len(current_chunk_comments))} comments, {current_chunk_chars} chars)...")
            chunk_text = COMMENT_SEPARATOR.join(current_chunk_comments)
            try:
                # Call the AI client's analyze method for this chunk
                chunk_result = ai_client.analyze(chunk_text, ANALYSIS_SYSTEM_PROMPT)
                all_analysis_results.append(chunk_result)
            except Exception as e:
                print(f"Error analyzing chunk {len(all_analysis_results) + 1}: {e}")
                # Append an error placeholder or skip, depending on desired handling
                all_analysis_results.append({"error": f"Chunk analysis failed: {e}"}) # Important for JSON merging

            # 2. Start a new chunk with the current comment
            current_chunk_comments = [comment]
            current_chunk_chars = comment_len
        else:
            # Add the comment to the current chunk
            if current_chunk_comments: # Add separator length if not the first comment
                 current_chunk_chars += separator_len
            current_chunk_chars += comment_len
            current_chunk_comments.append(comment)

        # Progress update (optional)
        if (i + 1) % 50 == 0:
            print(f"  ...processed comment {i+1}/{len(scraped_comments_list)}")


    # --- Process the final remaining chunk ---
    if current_chunk_comments:
        print(f"Processing final chunk {len(all_analysis_results) + 1} ({(len(current_chunk_comments))} comments, {current_chunk_chars} chars)...")
        chunk_text = COMMENT_SEPARATOR.join(current_chunk_comments)
        try:
            chunk_result = ai_client.analyze(chunk_text, ANALYSIS_SYSTEM_PROMPT)
            all_analysis_results.append(chunk_result)
        except Exception as e:
            print(f"Error analyzing final chunk {len(all_analysis_results) + 1}: {e}")
            all_analysis_results.append({"error": f"Final chunk analysis failed: {e}"})

    # --- Merge results from all chunks ---
    if not all_analysis_results:
        print("No analysis results were generated from chunks.")
        return None

    print(f"Merging results from {len(all_analysis_results)} chunks...")
    final_result = merge_analysis_results(all_analysis_results)

    return final_result