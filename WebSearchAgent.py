import requests
import json
import os
import logging
import time
import re
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup # type: ignore
import html2text # type: ignore
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataSynthesizer')

class RateLimiter:
    """Simple rate limiter to avoid overwhelming servers."""
    
    def __init__(self, calls_per_minute: int = 30):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0
        self.call_history = []
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        
        # Clean old entries from history (older than 1 minute)
        cutoff_time = current_time - 60
        self.call_history = [t for t in self.call_history if t > cutoff_time]
        
        # If we've made too many calls in the last minute, wait
        if len(self.call_history) >= self.calls_per_minute:
            wait_time = 60 - (current_time - self.call_history[0])
            if wait_time > 0:
                logger.info(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
        
        # Also ensure minimum interval between calls
        time_since_last = current_time - self.last_call
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
        
        # Record this call
        self.call_history.append(time.time())
        self.last_call = time.time()

class ContentExtractor:
    """Extract relevant content from different types of documentation sites."""
    
    def __init__(self):
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.body_width = 0  # Don't wrap lines
    
    def extract_from_docs_site(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extract content from documentation sites like readthedocs, mimesis docs."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove navigation, footer, sidebar elements
        for element in soup.find_all(['nav', 'footer', 'aside', 'header']):
            element.decompose()
        
        # Remove elements with common navigation classes
        for class_name in ['navigation', 'sidebar', 'toctree', 'footer', 'header-links']:
            for element in soup.find_all(class_=re.compile(class_name, re.I)):
                element.decompose()
        
        # Extract main content
        main_content = soup.find('main') or soup.find('div', class_=re.compile('content|main', re.I)) or soup
        
        # Extract code examples
        code_blocks = []
        for code_element in main_content.find_all(['code', 'pre']):
            code_text = code_element.get_text().strip()
            if len(code_text) > 10:  # Filter out small inline code
                code_blocks.append(code_text)
        
        # Extract text content
        text_content = self.html_converter.handle(str(main_content))
        
        # Clean up the text
        text_content = re.sub(r'\n\s*\n\s*\n', '\n\n', text_content)  # Remove excessive newlines
        text_content = text_content.strip()
        
        return {
            "content_type": "documentation",
            "text_content": text_content[:5000],  # Limit text content
            "code_examples": code_blocks[:10],     # Limit code examples
            "source_url": url
        }
    
    def extract_from_stackoverflow(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extract content from Stack Overflow answers."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find the accepted answer or top-voted answers
        answers = []
        answer_elements = soup.find_all('div', class_=re.compile('answer', re.I))
        
        for answer_element in answer_elements[:3]:  # Top 3 answers
            # Extract code blocks from this answer
            code_blocks = []
            for code_element in answer_element.find_all(['code', 'pre']):
                code_text = code_element.get_text().strip()
                if len(code_text) > 10:
                    code_blocks.append(code_text)
            
            # Extract answer text
            answer_text = answer_element.get_text().strip()
            
            if code_blocks or len(answer_text) > 50:
                answers.append({
                    "text": answer_text[:1000],  # Limit answer text
                    "code_examples": code_blocks
                })
        
        return {
            "content_type": "stackoverflow",
            "answers": answers,
            "source_url": url
        }
    
    def extract_from_github(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extract content from GitHub README or code files."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Check if it's a README file
        readme_content = soup.find('div', {'id': 'readme'}) or soup.find('article', class_=re.compile('markdown', re.I))
        
        if readme_content:
            # Extract code blocks from README
            code_blocks = []
            for code_element in readme_content.find_all(['code', 'pre']):
                code_text = code_element.get_text().strip()
                if len(code_text) > 10:
                    code_blocks.append(code_text)
            
            # Extract text content
            text_content = self.html_converter.handle(str(readme_content))
            text_content = re.sub(r'\n\s*\n\s*\n', '\n\n', text_content).strip()
            
            return {
                "content_type": "github_readme",
                "text_content": text_content[:3000],
                "code_examples": code_blocks[:8],
                "source_url": url
            }
        else:
            # It might be a code file
            code_content = soup.find('div', class_=re.compile('highlight', re.I))
            if code_content:
                code_text = code_content.get_text().strip()
                return {
                    "content_type": "github_code",
                    "code_content": code_text[:2000],
                    "source_url": url
                }
        
        return {
            "content_type": "github_other",
            "raw_content": soup.get_text()[:1000],
            "source_url": url
        }
    
    def extract_from_pypi(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extract content from PyPI package pages."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract package description
        description_element = soup.find('div', class_=re.compile('project-description', re.I))
        
        code_blocks = []
        text_content = ""
        
        if description_element:
            # Extract code examples
            for code_element in description_element.find_all(['code', 'pre']):
                code_text = code_element.get_text().strip()
                if len(code_text) > 10:
                    code_blocks.append(code_text)
            
            # Extract text content
            text_content = self.html_converter.handle(str(description_element))
            text_content = re.sub(r'\n\s*\n\s*\n', '\n\n', text_content).strip()
        
        return {
            "content_type": "pypi",
            "text_content": text_content[:3000],
            "code_examples": code_blocks[:5],
            "source_url": url
        }
    
    def extract_content(self, html_content: str, url: str) -> Dict[str, Any]:
        """Main method to extract content based on URL domain."""
        domain = urlparse(url).netloc.lower()
        
        try:
            if 'stackoverflow.com' in domain:
                return self.extract_from_stackoverflow(html_content, url)
            elif 'github.com' in domain:
                return self.extract_from_github(html_content, url)
            elif 'pypi.org' in domain:
                return self.extract_from_pypi(html_content, url)
            else:
                # Default to documentation site extraction
                return self.extract_from_docs_site(html_content, url)
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return {
                "content_type": "extraction_failed",
                "error": str(e),
                "source_url": url
            }

class WebSearchAgent:
    """
    Enhanced Web Search Agent with targeted site searching and content scraping.
    """
    
    def __init__(self, api_key: Optional[str] = None, cx: Optional[str] = None):
        """Initialize the Enhanced Web Search Agent."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.cx = cx or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        
        # Initialize components
        self.rate_limiter = RateLimiter(calls_per_minute=30)  # Conservative rate limit
        self.content_extractor = ContentExtractor()
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Validate credentials
        if not self.api_key:
            logger.warning("Google API key not provided")
        if not self.cx:
            logger.warning("Custom Search Engine ID not provided")
            
        logger.info("Enhanced Web Search Agent initialized")
    
    def create_targeted_queries(self, base_query: str, library_name: str) -> List[str]:
        """Create targeted search queries for specific sites."""
        queries = []
        
        # Documentation sites
        if library_name.lower() == 'mimesis':
            queries.append(f"site:mimesis.name {base_query}")
        
        queries.extend([
            f"site:readthedocs.io {library_name} {base_query}",
            f"site:stackoverflow.com python {library_name} {base_query}",
            f"site:github.com {library_name} {base_query} example",
            f"site:pypi.org {library_name}"
        ])
        
        # Fallback general query
        queries.append(f"python {library_name} {base_query} example usage")
        
        return queries
    
    def search_with_retry(self, query: str, max_retries: int = 3) -> Dict[str, Any]:
        """Perform search with retry logic and rate limiting."""
        for attempt in range(max_retries):
            try:
                # Apply rate limiting
                self.rate_limiter.wait_if_needed()
                
                # API URL and parameters
                url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    'q': query,
                    'key': self.api_key,
                    'cx': self.cx,
                    'num': 5
                }
                
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    wait_time = 60 * (attempt + 1)  # Exponential backoff
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Search API error: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Search attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))  # Exponential backoff
        
        return {"error": "All search attempts failed"}
    
    def scrape_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape content from a URL with rate limiting."""
        try:
            # Apply rate limiting
            self.rate_limiter.wait_if_needed()
            
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                # Extract content using the content extractor
                extracted = self.content_extractor.extract_content(response.text, url)
                logger.info(f"Successfully scraped content from {url}")
                return extracted
            else:
                logger.warning(f"Failed to scrape {url}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return None
    
    def enhanced_search_for_column(self, column_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform enhanced search for a column with content scraping."""
        column_name = column_analysis.get("column_name", "unknown")
        generation_strategy = column_analysis.get("generation_strategy", "")
        
        # Skip if not using library search strategy
        if generation_strategy != "library_search":
            return {
                "column_name": column_name,
                "search_required": False,
                "content": []
            }
        
        # Get original search queries and provider details
        search_queries = column_analysis.get("search_queries", [])
        provider_details = column_analysis.get("provider_details", {})
        library_name = provider_details.get("library", "")
        
        logger.info(f"Enhanced search for column {column_name} using library {library_name}")
        
        # Create enhanced targeted queries
        all_queries = []
        for query in search_queries[:2]:  # Limit original queries
            if library_name:
                targeted_queries = self.create_targeted_queries(query, library_name)
                all_queries.extend(targeted_queries[:3])  # Top 3 targeted queries per original query
            else:
                all_queries.append(query)
        
        # Perform searches and scrape content
        scraped_content = []
        processed_urls = set()  # Avoid duplicates
        
        for query in all_queries[:5]:  # Limit total queries
            logger.info(f"Searching: {query}")
            
            search_results = self.search_with_retry(query)
            
            if "error" in search_results:
                continue
            
            # Process search results
            for item in search_results.get("items", [])[:3]:  # Top 3 results per query
                url = item.get("link", "")
                
                if url in processed_urls:
                    continue
                
                processed_urls.add(url)
                
                # Scrape content from this URL
                content = self.scrape_content(url)
                if content:
                    # Add search context
                    content["search_query"] = query
                    content["search_title"] = item.get("title", "")
                    content["search_snippet"] = item.get("snippet", "")
                    scraped_content.append(content)
        
        return {
            "column_name": column_name,
            "search_required": True,
            "library_name": library_name,
            "queries_processed": len(all_queries),
            "urls_scraped": len(scraped_content),
            "content": scraped_content
        }
    
    def process_batch_enhanced_searches(self, ordered_columns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process enhanced searches for a batch of columns."""
        logger.info(f"Starting enhanced batch search for {len(ordered_columns)} columns")
        
        # Filter columns that need search
        columns_needing_search = [
            col for col in ordered_columns 
            if col.get("generation_strategy") == "library_search"
        ]
        
        logger.info(f"Found {len(columns_needing_search)} columns requiring library search")
        
        enhanced_results = {
            "timestamp": datetime.now().isoformat(),
            "total_columns": len(ordered_columns),
            "columns_searched": len(columns_needing_search),
            "results": {}
        }
        
        # Process each column
        for i, column in enumerate(columns_needing_search, 1):
            column_name = column.get("column_name", f"unknown_{i}")
            
            logger.info(f"Processing column {i}/{len(columns_needing_search)}: {column_name}")
            
            try:
                # Perform enhanced search
                search_result = self.enhanced_search_for_column(column)
                enhanced_results["results"][column_name] = search_result
                
                logger.info(f"Completed search for {column_name}: {search_result.get('urls_scraped', 0)} URLs scraped")
                
            except Exception as e:
                logger.error(f"Error processing column {column_name}: {str(e)}")
                enhanced_results["results"][column_name] = {
                    "column_name": column_name,
                    "error": str(e),
                    "content": []
                }
        
        # Save results
        output_path = "pipeline_run_outputs/web_search_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Enhanced batch search completed. Results saved to {output_path}")
        return enhanced_results

# Example usage
if __name__ == "__main__":
    # Initialize the enhanced agent
    agent = WebSearchAgent()
    
    # Load the ordered columns from the LLM Analysis component
    with open("ordered_columns.json", 'r', encoding='utf-8') as f:
        ordered_columns = json.load(f)
    
    # Process enhanced batch searches
    enhanced_results = agent.process_batch_enhanced_searches(ordered_columns)
    
    print(f"Enhanced search processing complete. Processed {enhanced_results['columns_searched']} columns.")