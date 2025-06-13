import json
import os
import logging
from typing import Dict, List, Any, Optional, Union
import google.generativeai as genai #type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataSynthesizer')

class CodeGenerator:
    """
    Component responsible for generating Python code for data synthesis
    using LLM (Gemini 2.5).
    """

    def __init__(self, api_key: str):
        """
        Initialize the Code Generator component.
        """
        self.api_key = api_key

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(
            #model_name="models/gemini-2.5-pro-preview-05-06"
            model_name="models/gemini-2.5-flash-preview-04-17"
        )

        logger.info(f"Code Generator initialized with {self.model.model_name}")

    def _gather_feedback(self, feedback: Optional[Dict[str, Any]]) -> str:
        """
        Compiles feedback from code review and execution/validation failures. Not being used.
        """
        feedback_parts = []
        feedback_parts.append("The previous attempt to generate and run the data synthesis code had issues.")

        if feedback and not feedback.get("pass", True) :
            feedback_parts.append("\n--- Code Review Feedback ---")
            if feedback.get("summary"):
                feedback_parts.append(f"Review Summary: {feedback['summary']}")

            critical_issues = feedback.get("critical_issues", [])
            if critical_issues:
                feedback_parts.append("Critical Issues Found:")
                for issue in critical_issues:
                    feedback_parts.append(
                        f"- Type: {issue.get('type', 'N/A')}\n"
                        f"  Description: {issue.get('description', 'N/A')}\n"
                        f"  Location: {issue.get('location', 'N/A')}\n"
                        f"  Recommendation: {issue.get('recommendation', 'N/A')}"
                    )

            non_critical_issues = feedback.get("non_critical_issues", [])
            if non_critical_issues:
                feedback_parts.append("Non-Critical Issues/Suggestions:")
                for issue in non_critical_issues:
                     feedback_parts.append(
                        f"- Type: {issue.get('type', 'N/A')}\n"
                        f"  Description: {issue.get('description', 'N/A')}\n"
                        f"  Location: {issue.get('location', 'N/A')}\n"
                        f"  Recommendation: {issue.get('recommendation', 'N/A')}"
                    )

        if feedback and not feedback.get("overall_success", True):
            feedback_parts.append("\n--- Code Execution/Validation Feedback ---")
            if feedback.get("summary"):
                 feedback_parts.append(f"Execution/Validation Summary: {feedback['summary']}")

            exec_details = feedback.get("execution", {})
            if not exec_details.get("success", True):
                feedback_parts.append("Execution Failed:")
                if exec_details.get("error"):
                    feedback_parts.append(f"  Error: {exec_details['error']}")
                if exec_details.get("traceback"):
                    feedback_parts.append(f"  Traceback:\n{exec_details['traceback']}")

            val_details = feedback.get("validation", {})
            if not val_details.get("success", True):
                feedback_parts.append("Data Validation Failed:")
                if val_details.get("summary"):
                     feedback_parts.append(f"  Validation Summary: {val_details['summary']}")
                failed_checks = [v for v in val_details.get("validations", []) if not v.get("passed")]
                if failed_checks:
                    feedback_parts.append("  Specific Validation Failures:")
                    for check in failed_checks[:5]: # Limit to first 5 for brevity
                        feedback_parts.append(
                            f"  - Check: {check.get('name', 'N/A')}\n"
                            f"    Description: {check.get('description', 'N/A')}\n"
                            f"    Details: {check.get('details', 'N/A')}"
                        )

        return "\n".join(feedback_parts)
    
    def generate_imports_setup_block(self,
                                     general_instructions: Dict[str, Any],
                                     search_results: Dict[str, Any],
                                     relevant_ordered_columns_data: str,
                                     feedback: Optional[str] = None,
                                     previous_code: Optional[str] = None,
                                     structured_critical_issues: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generates the imports and initial setup code block.
        """
        logger.info("Generating 'imports_setup' block.")
        section_name = "imports_setup"

        feedback_section = f"\n# Previous Attempt Feedback (for 'imports_setup' section):\n{feedback}\n" if feedback else ""
        previous_code_section = f"\n# Previous Code for 'imports_setup' section:\n```python\n{previous_code}\n```\n" if previous_code else ""

        structured_issues_str = json.dumps(structured_critical_issues, indent=2) if structured_critical_issues else "No single structured critical issue identified for this section, rely on general feedback."

        search_results_section = "### Search Results Information:\n\n"
        search_results_data = search_results.get("results", {}) 

        for column_name, result in search_results_data.items():
            if isinstance(result, dict) and result.get("search_required", False) and "content" in result:
                search_results_section += f"#### Search Results for {column_name}:\n\n"
                for content_entry in result["content"][:3]:
                    code_examples = content_entry.get("code_examples", [])
                    if code_examples:
                        search_results_section += "Code Examples:\n"
                        for example in code_examples:  
                            search_results_section += f"```python\n{example}\n```\n\n"

                    text_snippet = content_entry.get("text_content", "").strip()
                    if text_snippet:
                        search_results_section += "Additional Context:\n"
                        search_results_section += f"{text_snippet[:1000]}\n\n"

        prompt = f"""
You are an expert Python developer. Your task is to generate only the imports, logging setup, and constant definitions
section for a data synthesis script, and also also to fix and regenerate the section if error feedback is provided.

# Structured Critical Issues (JSON - This is the HIGHEST priority list to fix) (DO NOT modify parts of the code that are not related to these issues, only make necessary changes, keep all naming consistent):
{structured_issues_str}
{previous_code_section}

# Instructions for generating imports and setup:
- Include necessary imports (e.g., pandas, mimesis, random, datetime).
- Set up basic logging.
- Define constants for output file path, format, and number of rows from general_instructions.
- Use mimesis version 18.0.0.
- Do not include any comments.
- Do not include any functions or main logic.

# Instructions for fixing issues:
- Analyze the 'Previous Code', the 'Previous Attempt Feedback', and especially the 'Structured Critical Issues'.
- Your primary goal is to *identify and modify only the necessary parts* of the 'Previous Code' to resolve ALL issues in the feedback, with top priority to the 'Structured Critical Issues'.
- Do NOT rewrite entire sections or functions if only a small change is needed.
- Preserve parts of the 'Previous Code' that are correct and not related to the feedback.
- Output only the *complete, updated Python code block for this specific section*.

# My Analysis and Plan to Fix Issues:
# Based on the feedback and previous code, briefly outline your understanding of the main problem(s) and your step-by-step plan to fix them in this section.
# 1. Main problem from feedback: [e.g., "SyntaxError: invalid syntax in import statement"]
#    My plan: [e.g., "Correct the import statement for 'pandas' from 'import pandas asp d' to 'import pandas as pd'."]
# 2. (If another distinct problem)
#    My plan: [...]

Final Code Review and Self-Correction:
# Before outputting the Python code block for this section, perform a final, comprehensive review.
# Carefully check the generated (or regenerated) code against ALL the instructions and feedback provided above.
# Ensure that:
# - All explicit instructions for this section have been strictly followed.
# - All critical issues from 'Structured Critical Issues' have been definitively resolved. (if there are any)
# - All points from 'Previous Attempt Feedback' have been addressed. (if there are any)
# - The code is syntactically correct and there are no extraneous comments, unnecessary imports, or irrelevant logic. 
# - All naming are consistent with other sections.
# - The code is concise, efficient, and directly fulfills the requirements for THIS specific section.
# - If this is a regenerated block, confirm that only necessary changes were made and correct parts of the 'Previous Code' were preserved.
#
# Provide your self-assessment: Does the code block for this section meet all requirements and resolve all identified issues?

# General Instructions:
{json.dumps(general_instructions, indent=2)}

# Relevant Column Data (for inferring imports):
{relevant_ordered_columns_data}

# Search Results (for library usage context):
{search_results_section}

Generate only the Python code block.
"""
        print("================= Prompt Length ===============")
        print(len(prompt))
        response = self.model.generate_content(prompt)
        return self._extract_code_from_response(response.text)
    
    def generate_all_column_logic_block(self,
                                        all_column_requirements: str,
                                        search_results: Dict[str, Any],
                                        imports_block_content: str,
                                        feedback: Optional[str] = None,
                                        previous_code: Optional[str] = None,
                                        structured_critical_issues: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generates all column generation functions in a single block.
        The LLM is responsible for determining the internal order and handling dependencies.
        """
        logger.info("Generating 'all_column_logic' block.")
        section_name = "all_column_logic"

        feedback_section = f"\n# Previous Attempt Feedback (for 'all_column_logic' section):\n{feedback}\n" if feedback else ""
        previous_code_section = f"\n# Previous Code for 'all_column_logic' section:\n```python\n{previous_code}\n```\n" if previous_code else ""

        structured_issues_str = json.dumps(structured_critical_issues, indent=2) if structured_critical_issues else "No single structured critical issue identified for this section, rely on general feedback."

        search_results_section = "### Search Results Information:\n\n"
        search_results_data = search_results.get("results", {}) 

        for column_name, result in search_results_data.items():
            if isinstance(result, dict) and result.get("search_required", False) and "content" in result:
                search_results_section += f"#### Search Results for {column_name}:\n\n"
                for content_entry in result["content"][:3]:
                    code_examples = content_entry.get("code_examples", [])
                    if code_examples:
                        search_results_section += "Code Examples:\n"
                        for example in code_examples:  
                            search_results_section += f"```python\n{example}\n```\n\n"

                    text_snippet = content_entry.get("text_content", "").strip()
                    if text_snippet:
                        search_results_section += "Additional Context:\n"
                        search_results_section += f"{text_snippet[:1000]}\n\n"

        prompt = f"""
You are an expert Python developer. Your task is to generate only all necessary Python functions and helper logic
to generate data for ALL the specified columns and also fix and regenerate this section if error feedback is provided. This is the second single block of code generation.

# Structured Critical Issues (JSON - This is the HIGHEST priority list to fix) (DO NOT modify parts of the code that are not related to these issues, only make necessary changes, keep all naming consistent):
{structured_issues_str}
{previous_code_section}

#JUST context from preceding blocks, do not include imports and setup logic:
# Context from Imports and Setup:
```python
{imports_block_content}
```

# All Column Requirements:
{all_column_requirements}

# Search Results (for library usage context):
{search_results_section}

# Instructions for generation:
- Create a separate function for each column (e.g., `def generate_column_name(...)`).
- Use mimesis (version 18.0.0) or custom logic as specified. Do NOT use faker.
- Implement any patterns, constraints, or relationships mentioned in the requirements across columns.
- The functions should return a single value for their respective columns.
- If a column depends on another, ensure its function correctly calls or uses the output of the dependent column's function.
- You are responsible for inferring the correct internal generation order of functions based on columns' relationships (which column is more important and decides other column's value should have its generation function first, e.g., if `column_b` contains `column_a`, `generate_column_a()` should be called before `generate_column_b()` to satisfy column_a requirements).
- Do not include any comments.
- Do not include imports or main execution logic.

# Instructions for fixing and regeneration:
- Analyze the 'Previous Code', the 'Previous Attempt Feedback', and especially the 'Structured Critical Issues'.
- Your primary goal is to *identify and modify only the necessary parts* of the 'Previous Code' to resolve ALL issues in the feedback, with top priority to the 'Structured Critical Issues'.
- Do NOT rewrite entire sections or functions if only a small change is needed.
- Preserve parts of the 'Previous Code' that are correct and not related to the feedback.
- Output only the *complete, updated Python code block for this specific section*.

# My Analysis and Plan to Fix Issues:
# Based on the feedback and previous code, briefly outline your understanding of the main problem(s) and your step-by-step plan to fix them in this section.
# 1. Main problem from feedback: [e.g., "Validation failed for 'age' column due to negative values."]
#    My plan: [e.g., "In the `generate_age` function, ensure the random number generation is always positive, for example by using `abs()` or adjusting the range."]
# 2. (If another distinct problem)
#    My plan: [...]

Final Code Review and Self-Correction:
# Before outputting the Python code block for this section, perform a final, comprehensive review.
# Carefully check the generated (or regenerated) code against ALL the instructions and feedback provided above.
# Ensure that:
# - All explicit instructions for this section have been strictly followed.
# - All critical issues from 'Structured Critical Issues' have been definitively resolved. (if there are any)
# - All points from 'Previous Attempt Feedback' have been addressed. (if there are any)
# - The code is syntactically correct and there are no extraneous comments, unnecessary imports, or irrelevant logic.
# - All naming are consistent with other sections.
# - The code is concise, efficient, and directly fulfills the requirements for THIS specific section.
# - If this is a regenerated block, confirm that only necessary changes were made and correct parts of the 'Previous Code' were preserved.
#
# Provide your self-assessment: Does the code block for this section meet all requirements and resolve all identified issues?

Generate only the Python code block containing all column generation functions.
"""
        print("================= Prompt Length ===============")
        print(len(prompt))
        response = self.model.generate_content(prompt)
        return self._extract_code_from_response(response.text)
    
    def generate_main_orchestration_block(self,
                                          imports_block_content: str,
                                          all_column_logic_code: str, 
                                          data_structure_instructions: Dict[str, Any],
                                          original_column_order: List[str],
                                          requirements_content: str,
                                          feedback: Optional[str] = None,
                                          previous_code: Optional[str] = None,
                                          structured_critical_issues: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generates the main orchestration logic, including the data generation loop.
        """
        logger.info("Generating 'main_orchestration' block.")
        section_name = "main_orchestration"

        feedback_section = f"\n# Previous Attempt Feedback (for 'main_orchestration' section):\n{feedback}\n" if feedback else ""
        previous_code_section = f"\n# Previous Code for 'main_orchestration' section:\n```python\n{previous_code}\n```\n" if previous_code else ""

        structured_issues_str = json.dumps(structured_critical_issues, indent=2) if structured_critical_issues else "No single structured critical issue identified for this section, rely on general feedback."

        prompt = f"""
You are an expert Python developer. Your task is to generate only the main orchestration logic for data synthesis and to also fix and regenerate this section if error feedback is provided. This is the third single block of code generation.

# Structured Critical Issues (JSON - This is the HIGHEST priority list to fix) (DO NOT modify parts of the code that are not related to these issues, only make necessary changes, keep all naming consistent):
{structured_issues_str}
{previous_code_section}

# Instructions for fixing and regeneration:
- Analyze the 'Previous Code', the 'Previous Attempt Feedback', and especially the 'Structured Critical Issues'.
- Your primary goal is to *identify and modify only the necessary parts* of the 'Previous Code' to resolve ALL issues in the feedback, with top priority to the 'Structured Critical Issues'.
- Do NOT rewrite entire sections or functions if only a small change is needed.
- Preserve parts of the 'Previous Code' that are correct and not related to the feedback.
- Output only the *complete, updated Python code block for this specific section*.

# My Analysis and Plan to Fix Issues:
# Based on the feedback and previous code, briefly outline your understanding of the main problem(s) and your step-by-step plan to fix them in this section.
# 1. Main problem from feedback: [e.g., "Execution error: DataFrame columns are not in the correct order."]
#    My plan: [e.g., "After creating the DataFrame, re-index it using the `original_column_order` list provided, like `df = df[original_column_order]`."]
# 2. (If another distinct problem)
#    My plan: [...]

Final Code Review and Self-Correction:
# Before outputting the Python code block for this section, perform a final, comprehensive review.
# Carefully check the generated (or regenerated) code against ALL the instructions and feedback provided above.
# Ensure that:
# - All explicit instructions for this section have been strictly followed.
# - All critical issues from 'Structured Critical Issues' have been definitively resolved. (if there are any)
# - All points from 'Previous Attempt Feedback' have been addressed. (if there are any)
# - The code is syntactically correct and there are no extraneous comments, unnecessary imports, or irrelevant logic.
# - All naming are consistent with other sections.
# - The code is concise, efficient, and directly fulfills the requirements for THIS specific section.
# - If this is a regenerated block, confirm that only necessary changes were made and correct parts of the 'Previous Code' were preserved.
#
# Provide your self-assessment: Does the code block for this section meet all requirements and resolve all identified issues?

#JUST context from preceding blocks, do not include imports or column generation logic:
# Context from Imports and Setup:
```python
{imports_block_content}
```
# Context from All Column Generation Functions:
```python
{all_column_logic_code}
```

# Data Structure Instructions:
{json.dumps(data_structure_instructions, indent=2)}

# Original Column Order (for final DataFrame ordering):
{json.dumps(original_column_order, indent=2)}

# Full Requirements (for understanding overall generation flow):
```
{requirements_content}
```

# Instructions for generation:
- Implement the main loop to generate `num_rows` (from constants in imports_setup) of data.
- Call the appropriate column generation functions (e.g., `generate_column_name()`) in the specified order.
- Aggregate the generated data into a list of dictionaries, then convert to a pandas DataFrame.
- Ensure the final DataFrame columns are in the `original_column_order`.
- Do not include any comments.
- Do not include imports or file output logic.

Generate only the Python code block for the main orchestration.
"""
        print("================= Prompt Length ===============")
        print(len(prompt))
        response = self.model.generate_content(prompt)
        return self._extract_code_from_response(response.text)
    
    def generate_file_output_block(self,
                                   imports_block_content: str,
                                   all_column_logic_code: str, 
                                   main_orchestration_dataframe_variable: str,
                                   original_column_order: List[str],
                                   output_format: str,
                                   feedback: Optional[str] = None,
                                   previous_code: Optional[str] = None,
                                   structured_critical_issues: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generates the code for saving the final data to a file.
        """
        logger.info("Generating 'file_output' block.")

        feedback_section = f"\n# Previous Attempt Feedback (for 'file_output' section):\n{feedback}\n" if feedback else ""
        previous_code_section = f"\n# Previous Code for 'file_output' section:\n```python\n{previous_code}\n```\n" if previous_code else ""

        structured_issues_str = json.dumps(structured_critical_issues, indent=2) if structured_critical_issues else "No single structured critical issue identified for this section, rely on general feedback."

        prompt = f"""
You are an expert Python developer. Your task is to generate only the code for saving the final generated data and to also fix and regenerate this section if error feedback is provided. This is the final single block of code generation.

# Structured Critical Issues (JSON - This is the HIGHEST priority list to fix) (DO NOT modify parts of the code that are not related to these issues, only make necessary changes, keep all naming consistent):
{structured_issues_str}
{previous_code_section}

# Instructions for fixing and regeneration:
- Analyze the 'Previous Code', the 'Previous Attempt Feedback', and especially the 'Structured Critical Issues'.
- Your primary goal is to *identify and modify only the necessary parts* of the 'Previous Code' to resolve ALL issues in the feedback, with top priority to the 'Structured Critical Issues'.
- Do NOT rewrite entire sections or functions if only a small change is needed.
- Preserve parts of the 'Previous Code' that are correct and not related to the feedback.
- Output only the *complete, updated Python code block for this specific section*.

# My Analysis and Plan to Fix Issues:
# Based on the feedback and previous code, briefly outline your understanding of the main problem(s) and your step-by-step plan to fix them in this section.
# 1. Main problem from feedback: [e.g., "Saving to CSV includes the DataFrame index."]
#    My plan: [e.g., "Add the `index=False` parameter to the `df.to_csv()` call."]
# 2. (If another distinct problem)
#    My plan: [...]

Final Code Review and Self-Correction:
# Before outputting the Python code block for this section, perform a final, comprehensive review.
# Carefully check the generated (or regenerated) code against ALL the instructions and feedback provided above.
# Ensure that:
# - All explicit instructions for this section have been strictly followed.
# - All critical issues from 'Structured Critical Issues' have been definitively resolved. (if there are any)
# - All points from 'Previous Attempt Feedback' have been addressed. (if there are any)
# - The code is syntactically correct and there are no extraneous comments, unnecessary imports, or irrelevant logic.
# - All naming are consistent with other sections.
# - The code is concise, efficient, and directly fulfills the requirements for THIS specific section.
# - If this is a regenerated block, confirm that only necessary changes were made and correct parts of the 'Previous Code' were preserved.
#
# Provide your self-assessment: Does the code block for this section meet all requirements and resolve all identified issues?

#JUST context from preceding blocks, do not include imports or column generation logic:
# Context from Imports and Setup:
```python
{imports_block_content}
```
# Context from All Column Generation Functions:
```python
{all_column_logic_code}
```

# Instructions for generation:
- Assume a pandas DataFrame named `{main_orchestration_dataframe_variable}` exists and contains the generated data.
- Save this DataFrame to the `OUTPUT_FILE_PATH` (from constants in imports_setup) in `{output_format}` format.
- Ensure the columns in the output file are in the `original_column_order`.
- Do not include any comments.
- Do not include imports or data generation/orchestration logic.

# Original Column Order:
{json.dumps(original_column_order, indent=2)}

Generate only the Python code block for file output.
"""
        print("================= Prompt Length ===============")
        print(len(prompt))
        response = self.model.generate_content(prompt)
        return self._extract_code_from_response(response.text)

    def _extract_code_from_response(self, response_text: str) -> str:
        """
        Extract Python code from the LLM response text.
        """
        try:
            # Look for Python code blocks (```python...```)
            code_start = response_text.find("```python")
            if code_start >= 0:
                code_start += len("```python")
                code_end = response_text.find("```", code_start)
                if code_end >= 0:
                    return response_text[code_start:code_end].strip()

            code_start = response_text.find("```")
            if code_start >= 0:
                code_start += len("```")
                code_end = response_text.find("```", code_start)
                if code_end >= 0:
                    return response_text[code_start:code_end].strip()

            logger.warning("No code block markers found in response, returning full text")
            return response_text.strip()

        except Exception as e:
            logger.error(f"Error extracting code from response: {str(e)}")
            logger.debug(f"Response text: {response_text[:200]}...")  # Log first 500 chars for debugging
            return response_text.strip()
        
    def _extract_text_from_response(self, response_text: str) -> str:
        """
        Extract Python code from the LLM response text.
        """
        try:
            # Look for Python code blocks (```python...```)
            code_start = response_text.find("```python")
            if code_start >= 0:
                code_start += len("```python")
                code_end = response_text.find("```", code_start)
                if code_end >= 0:
                    return response_text[:code_start].strip()

            code_start = response_text.find("```")
            if code_start >= 0:
                code_start += len("```")
                code_end = response_text.find("```", code_start)
                if code_end >= 0:
                    return response_text[:code_start].strip()

            logger.warning("No code block markers found in response, returning full text")
            return response_text.strip()

        except Exception as e:
            logger.error(f"Error extracting code from response: {str(e)}")
            logger.debug(f"Response text: {response_text[:200]}...")  # Log first 500 chars for debugging
            return response_text.strip()

    def save_generated_code(self, code: str, output_path: str = "pipeline_run_outputs/generated_data_script.py") -> str:
        """
        Save the generated code to a file.
        """
        logger.info(f"Saving generated code to {output_path}")

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(code)
            logger.info(f"Code successfully saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving generated code: {str(e)}")
            raise

if __name__ == "__main__":
    api_key = os.getenv("GOOGLE_API_KEY")

    generator = CodeGenerator(api_key)