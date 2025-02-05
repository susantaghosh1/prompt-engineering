import re

def chunk_markdown(markdown_content):
    """
    Chunks a markdown file based on headers and subheaders, associating
    content, headers, and page numbers with each chunk.

    Args:
        markdown_content: The markdown content as a string.

    Returns:
        A list of dictionaries, where each dictionary represents a chunk and
        contains the following keys:
            "content": The content of the chunk.
            "headers": A list of headers associated with the chunk.
            "page_numbers": A list of page numbers the chunk spans.
    """

    pages = markdown_content.split("---")
    chunks = []
    current_headers = []  # Keep track of the current header hierarchy
    current_page = 1

    for page in pages:
        lines = page.splitlines()
        for line in lines:
            header_match = re.match(r"^(#+) (.+)$", line)  # Matches headers (H1 to H6)
            if header_match:
                level = len(header_match.group(1))  # Header level (1 for H1, 2 for H2, etc.)
                header_text = header_match.group(2).strip()

                # Adjust current_headers based on the new header level
                current_headers = current_headers[:level - 1]  # Remove lower-level headers
                current_headers.append(header_text)  # Add the current header

                # Start a new chunk
                chunks.append({
                    "content": "",
                    "headers": list(current_headers),  # Copy the headers
                    "page_numbers": [current_page]
                })
            elif chunks:  # If a chunk is already started, add content to it
                chunks[-1]["content"] += line + "\n"
                if current_page not in chunks[-1]["page_numbers"]:
                    chunks[-1]["page_numbers"].append(current_page)
            elif not chunks and line.strip(): # handle the content before the first header
                chunks.append({
                    "content": line + "\n",
                    "headers": [],
                    "page_numbers": [current_page]
                })

        current_page += 1

    return chunks



# Example usage (assuming your markdown is in 'input.md'):
with open("input.md", "r", encoding="utf-8") as f:
    markdown_content = f.read()

chunks = chunk_markdown(markdown_content)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:")
    print(f"  Headers: {chunk['headers']}")
    print(f"  Page Numbers: {chunk['page_numbers']}")
    print(f"  Content:\n{chunk['content']}")
    print("-" * 20)

