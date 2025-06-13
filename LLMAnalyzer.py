import json
import os
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import google.generativeai as genai #type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataSynthesizer')

class LLMAnalyzer:
    
    def __init__(self, api_key: str, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the LLM Analyzer component.
        
        Args:
            api_key: Google API key for Gemini
            max_retries: Maximum number of retries for failed column analysis
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Initialize the model (Gemini 2.5 Flash - pro too slow)
        self.model = genai.GenerativeModel(
            model_name="models/gemini-2.5-flash-preview-04-17"
            #model_name="models/gemini-2.5-pro-preview-05-06"
        )
        
        logger.info(f"LLM Analyzer initialized with {self.model.model_name} model (max_retries: {max_retries})")
    
    def analyze_column(self, 
                      column_info: Dict[str, Any], 
                      table_info: Dict[str, Any],
                      column_relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a single column to determine its data generation strategy with retry mechanism.
        """
        column_name = column_info.get('Column_name')
        logger.info(f"Analyzing column: {column_name}")
        
        # Construct the prompt for the LLM
        prompt = self._construct_column_analysis_prompt(column_info, table_info, column_relationships)
        
        # Retry mechanism for LLM analysis
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Attempt {attempt}/{self.max_retries} for column {column_name}")
                
                # Call the LLM with the prompt
                response = self.model.generate_content(prompt)
                
                # Parse the response
                analysis_result = self._parse_column_analysis_response(response.text)
                
                # Check if parsing was successful (not a fallback result)
                if not analysis_result.get('column_name', '').startswith('unknown'):
                    logger.info(f"Successfully analyzed column {column_name} on attempt {attempt}")
                    return analysis_result
                else:
                    logger.warning(f"Parsing failed for column {column_name} on attempt {attempt}")
                    if attempt < self.max_retries:
                        logger.info(f"Retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        logger.error(f"All {self.max_retries} attempts failed for column {column_name}")
                        return analysis_result
                        
            except Exception as e:
                logger.error(f"Error analyzing column {column_name} on attempt {attempt}: {str(e)}")
                if attempt < self.max_retries:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    logger.error(f"All {self.max_retries} attempts failed for column {column_name}")
                    # Return fallback result after all retries exhausted
                    return {
                        "column_name": column_name or "unknown",
                        "priority": "low",
                        "generation_strategy": "custom_pattern",
                        "provider_details": {},
                        "pattern_analysis": f"Failed to analyze after {self.max_retries} attempts: {str(e)}",
                        "search_queries": [],
                        "custom_generation_approach": "Manual analysis required due to repeated failures.",
                        "constraints": ["Manual verification needed"],
                        "dependencies": []
                    }
        
        return {
            "column_name": column_name or "unknown",
            "priority": "low",
            "generation_strategy": "custom_pattern",
            "provider_details": {},
            "pattern_analysis": "Unexpected failure in retry mechanism",
            "search_queries": [],
            "custom_generation_approach": "Manual analysis required.",
            "constraints": ["Manual verification needed"],
            "dependencies": []
        }
    
    def _construct_column_analysis_prompt(self, 
                                        column_info: Dict[str, Any], 
                                        table_info: Dict[str, Any],
                                        column_relationships: List[Dict[str, Any]]) -> str:
        """
        Construct a detailed prompt for the LLM to analyze a column.
        """
        column_name = column_info.get('Column_name', 'Unknown')
        description = column_info.get('Description', 'No description available')
        key_type = column_info.get('Key_type', 'null')
        data_type = column_info.get('data_type', 'unknown')
        
        # Extract sample data if available
        sample_data = column_info.get('sample_data', [])
        sample_data_str = "\n".join([f"- {sample}" for sample in sample_data[:10]])
        
        # Extract basic stats if available
        stats_sections = []
        if 'null_percentage' in column_info:
            stats_sections.append(f"Null percentage: {column_info['null_percentage']:.2f}%")
        if 'num_unique' in column_info:
            stats_sections.append(f"Number of unique values: {column_info['num_unique']}")
        
        # Add type-specific stats
        if data_type in ['integer', 'float']:
            if 'min_val' in column_info and 'max_val' in column_info:
                stats_sections.append(f"Range: {column_info['min_val']} to {column_info['max_val']}")
            if 'average' in column_info:
                stats_sections.append(f"Average: {column_info['average']}")
            if 'positive_percentage' in column_info:
                stats_sections.append(f"Positive values: {column_info['positive_percentage']:.2f}%")
            if 'zero_percentage' in column_info:
                stats_sections.append(f"Zero values: {column_info['zero_percentage']:.2f}%")
        
        # Add categorical distribution if available
        distribution_section = ""
        if data_type == 'categorical' and 'value_distribution' in column_info:
            distribution_section = "Value distribution:\n"
            for value, percentage in column_info['value_distribution'].items():
                distribution_section += f"- {value}: {percentage:.2f}%\n"
        
        # Format relationships information
        relationships_section = ""
        if column_relationships:
            relationships_section = "Column relationships:\n"
            for rel in column_relationships:
                rel_type = rel.get('type', 'unknown')
                columns = rel.get('columns', [])
                formula = rel.get('formula', '')
                description = rel.get('description', '')
                
                if column_name in columns:
                    relationships_section += f"- Type: {rel_type}\n"
                    relationships_section += f"  Columns: {', '.join(columns)}\n"
                    if formula:
                        relationships_section += f"  Formula: {formula}\n"
                    if description:
                        relationships_section += f"  Description: {description}\n"
        
        # Construct the full prompt
        prompt = f"""
You are a Vietnamese Banker and an expert data analyst and Python programmer. Your task is to analyze metadata for a database column 
and determine the optimal strategy for generating synthetic data that matches its patterns and characteristics.

TABLE INFORMATION:
Name: {table_info.get('schema', 'Unknown')}.{table_info.get('name', 'Unknown')}
Description: {table_info.get('description', 'No description available')}

COLUMN INFORMATION:
Name: {column_name}
Description: {description}
Key Type: {key_type}
Data Type: {data_type}

STATISTICS:
{chr(10).join(stats_sections)}

SAMPLE DATA:
{sample_data_str}

{distribution_section}
{relationships_section}

Based on this information, please:

1. Analyze the patterns in the sample data as a Vietnamese Banker and an expert data analyst and Python programmer
2. Determine if this column should be high priority for generation (e.g., primary key, required for relationships)
3. Recommend the best strategy for generating synthetic data for this column (IMPORTANT: DO NOT USE faker library, use libraries that are available on Windows, Python 3) (Note: mimesis library does not support Vietnamese Locale yet, so for Vietnamese related data, you should use specialized libraries (like vn_fullname_generator) or custom patterns):
   - For common data types, like number, date, time, etc., you don't need to use mimesis, you can use custom patterns or specialized libraries. 
   - If you suggest using standard providers from the mimesis library (Version 18.0.0), you MUST set the 'generation_strategy' to "library_search" to confirm current usage and syntax. Populate 'provider_details' with the library ("mimesis") and specific provider (e.g., "datetime.month"). Crucially, also provide relevant 'search_queries' to find current Mimesis usage examples for this specific data type/provider.
   - For more complex patterns where mimesis might not suffice, if you identify specialized libraries (e.g., vn_fullname_generator), set 'generation_strategy' to "library_search". Populate 'provider_details' with the library name. Provide specific 'search_queries' to find current usage examples of this specialized library for the column's task. (libraries should preferably be on PYPI).
   - If no suitable library exists or is identified, recommend a "custom_pattern" strategy. In this case, 'search_queries' should be an empty list unless there's a generic algorithmic pattern worth searching for.
   - Try not to sample exact values if found a clear pattern (you can sample part of it (e.g., "TUAN24" → can sample "TUAN")).
4. Identify specific constraints or requirements for generating this data (e.g., data should follow exactly the formats of original (upper case, vietnamese with no special characters), data should follow statistical distribution of original (% null, etc.))
5. The results would be used for an LLM prompt, so keep the results to the point

Return your analysis in carefully in EXACT JSON format with the following structure:
{{
  "column_name": "name of the column",
  "priority": "high/medium/low",
  "generation_strategy": "library_search/custom_pattern",
  "provider_details": {{
    "library": "library name if applicable",
    "provider": "provider name if applicable"
  }},
  "pattern_analysis": "detailed description of observed patterns",
  "search_queries": ["ALWAYS provide a list of search queries if 'generation_strategy' is 'library_search' (i.e., whenever any library like mimesis (Version 18.0.0) or a specialized one is recommended). These queries should be precise, focused, help to find the up-to-date usage examples for the library for this specific column's data type/provider, focus on searching on reliable sites (library's doc, github, PYPI, NO stackoverflow). Provide an empty list if 'generation_strategy' is 'custom_pattern' and no library is involved or if no specific query is beneficial."],
  "custom_generation_approach": "detailed approach for custom generation if 'generation_strategy' is 'custom_pattern'",
  "constraints": ["list of constraints to respect when generating data"],
  "dependencies": ["list of columns this depends on or that depend on this"]
}}

Focus only on determining the optimal generation strategy and requirements based on the provided information.
"""
        return prompt
    
    def _parse_column_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract structured analysis results.
        """
        try:
            # Attempt to find and extract JSON content.
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')

            if json_start >= 0 and json_end >= json_start:
                json_str = response_text[json_start : json_end + 1]
                logger.debug(f"Extracted potential JSON string: {json_str[:500]}...") # Log extracted string

                # Attempt to load the extracted string as JSON
                analysis = json.loads(json_str)
                logger.info("Successfully parsed LLM response as JSON.")
                return analysis
            else:
                logger.error("No JSON object found in LLM response.")
                logger.debug(f"Full response text: {response_text}")
                return {
                    "column_name": "unknown - parsing failed",
                    "priority": "low", 
                    "generation_strategy": "custom_pattern", 
                    "pattern_analysis": f"Failed to parse LLM response: No JSON object found. Response: {response_text[:100]}...",
                    "search_queries": [],
                    "custom_generation_approach": "Manual analysis required due to parsing failure.",
                    "constraints": ["Manual verification needed"],
                    "dependencies": []
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM response as JSON: {str(e)}")
            logger.debug(f"Response text: {response_text}")
            return {
                "column_name": "unknown",
                "priority": "low",
                "generation_strategy": "custom_pattern",
                "pattern_analysis": "Failed to parse LLM response: JSON decode error",
                "constraints": []
            }
    
    def analyze_all_columns(self, 
                          enriched_metadata: Dict[str, Any], 
                          column_relationships: Dict[str, List]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze all columns in the enriched metadata and determine generation strategies.
        """
        logger.info("Starting analysis of all columns")
        
        table_analyses = {}
        
        # Process each table in the metadata
        for table_name, table_info in enriched_metadata.items():
            logger.info(f"Analyzing table: {table_name}")
            
            if 'columns' not in table_info:
                logger.warning(f"No columns defined for table {table_name}")
                continue
            
            column_analyses = []
            
            # Process each column in the table
            for column_info in table_info['columns']:
                column_name = column_info.get('Column_name')
                
                # Get relationships for this column
                column_rels = []
                for rel in column_relationships.get('relationships', []):
                    if column_name in rel.get('columns', []):
                        column_rels.append(rel)
                
                # Analyze the column
                analysis = self.analyze_column(column_info, table_info, column_rels)
                column_analyses.append(analysis)
            
            table_analyses[table_name] = column_analyses
            logger.info(f"Completed analysis for table {table_name}")
        
        logger.info("Completed analysis of all columns")
        return table_analyses
    
    def determine_generation_order(self, table_analyses: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Determine the optimal order for generating columns based on priorities and dependencies.
        """
        logger.info("Determining optimal generation order")
        
        all_columns = []
        
        # Flatten the table_analyses structure
        for table_name, column_analyses in table_analyses.items():
            for analysis in column_analyses:
                all_columns.append({
                    "table_name": table_name,
                    **analysis
                })
        
        # Sort columns by priority and dependencies
        # High priority columns first
        high_priority = [col for col in all_columns if col.get('priority') == 'high']
        medium_priority = [col for col in all_columns if col.get('priority') == 'medium']
        low_priority = [col for col in all_columns if col.get('priority') == 'low']
        
        # Start with high priority columns
        ordered_columns = high_priority.copy()
        
        # Add medium priority columns that depend on already included columns
        remaining = medium_priority.copy()
        while remaining:
            added_this_round = False
            for col in remaining[:]:
                dependencies = col.get('dependencies', [])
                already_included = [c.get('column_name') for c in ordered_columns]
                
                # If all dependencies are already included, add this column
                if all(dep in already_included for dep in dependencies):
                    ordered_columns.append(col)
                    remaining.remove(col)
                    added_this_round = True
            
            if not added_this_round:
                logger.warning("Possible circular dependencies detected when ordering columns")
                ordered_columns.extend(remaining)
                break
        
        ordered_columns.extend(low_priority)
        
        logger.info(f"Determined generation order for {len(ordered_columns)} columns")
        return ordered_columns
    
    def generate_requirements_document(self, ordered_columns: List[Dict[str, Any]]) -> str:
        """
        Generate a requirements document for column generation.
        """
        logger.info("Generating requirements document")
        
        requirements = []
        
        for idx, col in enumerate(ordered_columns, 1):
            column_name = col.get('column_name', 'Unknown')
            priority = col.get('priority', 'Unknown')
            table_name = col.get('table_name', 'Unknown')
            strategy = col.get('generation_strategy', 'Unknown')
            pattern = col.get('pattern_analysis', 'No pattern analysis available')
            
            req_text = f"## {idx}. Column '{column_name}' (Table: {table_name})\n\n"
            req_text += f"- Priority: {priority}\n"
            req_text += f"- Generation strategy: {strategy}\n"
            req_text += f"- Pattern analysis: {pattern}\n"
            
            # Add provider details if available
            if 'provider_details' in col and col['provider_details']:
                provider = col['provider_details'].get('provider', 'N/A')
                library = col['provider_details'].get('library', 'N/A')
                req_text += f"- Recommended provider: {provider} from {library}\n"
            
            # Add constraints
            constraints = col.get('constraints', [])
            if constraints:
                req_text += "- Constraints:\n"
                for constraint in constraints:
                    req_text += f"  - {constraint}\n"
            
            # Add dependencies
            dependencies = col.get('dependencies', [])
            if dependencies:
                req_text += "- Dependencies:\n"
                for dep in dependencies:
                    req_text += f"  - {dep}\n"
            
            # Add custom generation approach if applicable
            if strategy == 'custom_pattern' and 'custom_generation_approach' in col:
                approach = col.get('custom_generation_approach', 'No detailed approach specified')
                req_text += f"- Custom generation approach: {approach}\n"
            
            # Add search queries if applicable
            if strategy == 'library_search' and 'search_queries' in col:
                queries = col.get('search_queries', [])
                if queries:
                    req_text += "- Search queries:\n"
                    for query in queries:
                        req_text += f"  - \"{query}\"\n"
            
            requirements.append(req_text)
        
        requirements_doc = "\n".join(requirements)

        # Header
        header = "# Data Generation Requirements Document\n\n"
        header += "This document outlines the requirements for generating synthetic data for each column, "
        header += "listed in the order they should be generated.\n\n"
        
        requirements_doc = header + requirements_doc
        
        logger.info("Requirements document generated successfully")
        return requirements_doc
    
    def analyze_metadata(self, 
                        enriched_metadata: Dict[str, Any],
                        column_relationships: Dict[str, List]) -> Tuple[List[Dict[str, Any]], str]:
        """
        Main method to analyze metadata and determine generation strategies and order.
        """
        logger.info("Starting metadata analysis")
        
        # Analyze all columns
        table_analyses = self.analyze_all_columns(enriched_metadata, column_relationships)
        
        # Determine generation order
        ordered_columns = self.determine_generation_order(table_analyses)
        
        # Generate requirements document
        requirements_doc = self.generate_requirements_document(ordered_columns)
        
        # # Save requirements document to file
        # output_path = "column_generation_requirements.txt"
        # with open(output_path, 'w', encoding='utf-8') as f:
        #     f.write(requirements_doc)
        
        logger.info(f"Analysis complete.")
        return ordered_columns, requirements_doc

if __name__ == "__main__":
    api_key = os.getenv("GOOGLE_API_KEY")
    
    analyzer = LLMAnalyzer(api_key)
    
    with open("enriched_metadata.json", 'r', encoding='utf-8') as f:
        enriched_metadata = json.load(f)
    
    with open("column_relationships.json", 'r', encoding='utf-8') as f:
        column_relationships = json.load(f)

    ordered_columns, requirements_doc = analyzer.analyze_metadata(
        enriched_metadata, column_relationships
    )
    
    with open("ordered_columns.json", 'w', encoding='utf-8') as f:
        json.dump(ordered_columns, f, ensure_ascii=False, indent=2)
    
    print("LLM Analysis complete.")