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

# Get supported languages from environment variables and parse JSON array
try:
    SUPPORTED_LANGUAGES = json.loads(os.getenv("SUPPORTED_LANGUAGES", "[]"))
    if not SUPPORTED_LANGUAGES:
        raise ValueError("SUPPORTED_LANGUAGES environment variable is empty")
except json.JSONDecodeError:
    # Try comma-separated format as fallback
    SUPPORTED_LANGUAGES = [lang.strip() for lang in os.getenv("SUPPORTED_LANGUAGES", "").split(",") if lang.strip()]
    if not SUPPORTED_LANGUAGES:
        raise ValueError("SUPPORTED_LANGUAGES environment variable is not set or is empty")

def create_directory_structure():
    # Create base directory for translations
    base_dir = 'output'
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)
    
    for lang_code in SUPPORTED_LANGUAGES:
        lang_code = lang_code.strip().strip('"')
        folder_name = f'values-{lang_code}'
        os.makedirs(os.path.join(base_dir, folder_name))
    
    return base_dir
    
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def escape_quotes_outside_cdata(text):
    """
    Escapes single quotes in a string.
    """
    pattern = r"(')"
    def replacer(match):
        return r"\'"

    return re.sub(pattern, replacer, text, flags=re.DOTALL)

async def translate_file(source_file, target_language, llm, provider):
    # Read the entire file
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()
    app_name = os.getenv("APP_NAME", "Android").lower()
    num_tokens = num_tokens_from_string(content,"cl100k_base")
    print(f"Input Tokens : {num_tokens}")
    prompt = f"""You are a professional application translator. Translate this Android strings.xml file to language which has ISO code: {target_language}, in the context of {app_name} application.

Important rules:
1. DO NOT translate text marked with translatable="false"
2. Preserve all XML tags, attributes, and structure exactly
3. Preserve all entities like &appname; and &author;
4. Only translate the text values enclosed within the XML tags
5. Keep all special characters and formatting
6. Translate the text to the best of your ability, do not make up translations, be careful with phrase verbs.

Here's the file to translate:

{content}

Translated file:"""

    try:
        # Run the LLM invocation in a thread pool to prevent blocking
        with ThreadPoolExecutor() as executor:
            response = await asyncio.get_event_loop().run_in_executor(
                executor, lambda: llm.invoke(prompt)
            )
        
        if provider == "gemini":
            response = response.content
        
        # Replace single quotes with escaped single quotes
        translated_content = re.sub(r'```xml\s*|\s*```', '', response, flags=re.DOTALL)
        translated_content = escape_quotes_outside_cdata(translated_content)
        return translated_content
        
    except Exception as e:
        print(f"Error in translation for {target_language}: {e}")
        return content

async def translate_language(source_file, lang_code, base_dir):
    # Clean the language code
    lang_code = lang_code.strip().strip('"')
    print(f"\nStarting translation for {lang_code}...")

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

    # Translate the entire file
    translated_content = await translate_file(source_file, lang_code, llm, provider)

    # Save translated file
    folder_name = f'values-{lang_code}'
    output_file = os.path.join(base_dir, folder_name, 'strings.xml')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(translated_content)

    print(f"Completed translation for {lang_code}")

async def translate_strings_file(source_file):
    base_dir = create_directory_structure()
    
    # Create tasks for all languages
    tasks = [
        translate_language(source_file, lang_code, base_dir)
        for lang_code in SUPPORTED_LANGUAGES
    ]
    
    # Run all translations concurrently
    await asyncio.gather(*tasks)
    
    # Create zip file
    shutil.make_archive('ouput', 'zip', base_dir)
    print("\nTranslation completed! Files are saved in 'output.zip'")



if __name__ == '__main__':
    # Run the async main function
    asyncio.run(translate_strings_file('strings.xml'))