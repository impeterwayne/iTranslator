from langchain_openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import json
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tiktoken
import re

load_dotenv()

TOKEN_LIMIT = 8000 
try:
    SUPPORTED_LANGUAGES = json.loads(os.getenv("SUPPORTED_LANGUAGES", "[]"))
    if not SUPPORTED_LANGUAGES:
        raise ValueError("SUPPORTED_LANGUAGES environment variable is empty")
except json.JSONDecodeError:
    SUPPORTED_LANGUAGES = [lang.strip() for lang in os.getenv("SUPPORTED_LANGUAGES", "").split(",") if lang.strip()]
    if not SUPPORTED_LANGUAGES:
        raise ValueError("SUPPORTED_LANGUAGES environment variable is not set or is empty")

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    except Exception as e:
        print(f"Error getting token count: {e}")
        # Fallback to a rough estimate if tiktoken fails
        return len(string) // 4

def create_directory_structure():
    """Creates the output directory structure for the translated files."""
    base_dir = 'output'
    if os.path.exists(base_dir):
        print("Removing existing output directory.")
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)
    
    for lang_code in SUPPORTED_LANGUAGES:
        lang_code = lang_code.strip().strip('"')
        folder_name = f'values-{lang_code}'
        os.makedirs(os.path.join(base_dir, folder_name))
    
    return base_dir

def escape_quotes_outside_cdata(text):
    """
    Escapes single quotes in a string.
    """
    pattern = r"(')"
    def replacer(match):
            return r"\'"

    return re.sub(pattern, replacer, text, flags=re.DOTALL)

async def translate_chunk(chunk, target_language, llm, provider):
    """Translates a single chunk of XML content."""
    app_name = os.getenv("APP_NAME", "this application")
    
    prompt = f"""You are a professional XML translator for Android applications.
Translate the text content within the following XML snippet to the language with ISO code: '{target_language}'.

**Critical Rules:**
1.  **Preserve Structure:** Do NOT alter any XML tags, attributes (like `name` or `translatable="false"`), or placeholders (like `%1$s`, `%d`).
2.  **Respect Non-Translatable Content:** DO NOT translate text marked with `translatable="false"`
3.  **Translate Content Only:** Only translate the text values enclosed within the XML tags.
4.  **Entities:** Keep all named entities (e.g., `&amp;`, `&lt;`, `&copy;`) as they are.
5.  **Context:** The translation is for the '{app_name}' application.
6.  **Output XML Only:** Your response MUST contain ONLY the translated XML content. Do not include any extra text, explanations, apologies, or markdown formatting like ` ```xml `.
**XML Snippet to Translate:**
```xml
{chunk}
```

**Translated XML Snippet:**"""

    try:
        # Use a thread pool for the blocking LLM call
        with ThreadPoolExecutor() as executor:
            response = await asyncio.get_event_loop().run_in_executor(
                executor, lambda: llm.invoke(prompt)
            )
        
        # Extract content based on provider
        if provider == "gemini":
            translated_content = response.content
        else:
            translated_content = str(response)

        # Clean up the response, removing markdown code blocks if present
        cleaned_content = re.sub(r'```xml\s*|\s*```', '', translated_content, flags=re.DOTALL).strip()

        # Aggressively find the start of the first XML tag and strip everything before it.
        first_tag_index = cleaned_content.find('<')
        if first_tag_index > 0:
            cleaned_content = cleaned_content[first_tag_index:]
        
        cleaned_content = escape_quotes_outside_cdata(cleaned_content)

        return cleaned_content.strip()
        
    except Exception as e:
        print(f"An error occurred during translation of a chunk for {target_language}: {e}")
        # Return the original chunk on error to prevent data loss
        return chunk

def chunk_xml_file(source_file: str, encoding_name: str) -> list[str]:
    """
    Reads an XML file, extracts string resources, and groups them into chunks
    based on the token limit.
    """
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find all <string> and <string-array> elements.
    # This is more robust for handling nested tags within string arrays.
    resources = re.findall(r'(<plurals .*?</plurals>|<string-array .*?</string-array>|<string .*?</string>)', content, re.DOTALL)
    
    if not resources:
        print("Warning: No string resources found in the source file.")
        return []

    chunks = []
    current_chunk = ""
    current_tokens = 0

    for res in resources:
        res_tokens = num_tokens_from_string(res, encoding_name)
        
        if current_tokens + res_tokens > TOKEN_LIMIT and current_chunk:
            chunks.append(current_chunk)
            current_chunk = ""
            current_tokens = 0
            
        current_chunk += res + "\n"
        current_tokens += res_tokens

    if current_chunk:
        chunks.append(current_chunk)
        
    print(f"File was split into {len(chunks)} chunks.")
    return chunks

async def translate_language(source_file, lang_code, base_dir):
    """
    Manages the translation process for a single language, including chunking
    and file reconstruction.
    """
    lang_code = lang_code.strip().strip('"')
    print(f"\nStarting translation for {lang_code}...")

    # Create LLM client for this language task
    provider = os.getenv("TRANSLATION_PROVIDER", "openai").lower()
    llm = None
    if provider == "openai":
        llm = OpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=16000,
            temperature=0.5,
        )
    elif provider == "gemini":
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            max_output_tokens=65536,
            temperature=0.1,
            model_kwargs={
                "generation_config": {
                    "thinking_config": {"thinking_budget": 0}
                }
            })
    else:
        raise ValueError(f"Unknown TRANSLATION_PROVIDER: {provider}")


    # Chunk the source file
    encoding_name = "cl100k_base"
    chunks = chunk_xml_file(source_file, encoding_name)
    
    if not chunks:
        print(f"Skipping {lang_code} as no content was chunked.")
        return

    # Translate all chunks concurrently
    tasks = [translate_chunk(chunk, lang_code, llm, provider) for chunk in chunks]
    translated_chunks = await asyncio.gather(*tasks)
    
    # Reconstruct the full translated content
    full_translated_content = "\n".join(translated_chunks)
    
    # Create the final XML file structure
    final_xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<resources>\n'
        f'{full_translated_content}\n'
        '</resources>'
    )

    # Save the translated file
    folder_name = f'values-{lang_code}'
    output_file = os.path.join(base_dir, folder_name, 'strings.xml')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_xml)

    print(f"Completed translation for {lang_code}")

async def main(source_file):
    """Main function to orchestrate the translation process."""
    if not os.path.exists(source_file):
        print(f"Error: Source file not found at '{source_file}'")
        return

    base_dir = create_directory_structure()
    
    # Create and run tasks for all languages concurrently
    tasks = [
        translate_language(source_file, lang_code, base_dir)
        for lang_code in SUPPORTED_LANGUAGES
    ]
    await asyncio.gather(*tasks)
    
    # Create a zip archive of the output
    shutil.make_archive('output', 'zip', base_dir)
    print("\n✅ Translation process completed! Files are saved in 'output.zip'")

if __name__ == '__main__':
    source_strings_file = 'strings.xml'
    asyncio.run(main(source_strings_file))
