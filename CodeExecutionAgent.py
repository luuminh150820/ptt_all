import os
import sys
import traceback
import json
import csv
import logging
import subprocess
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Union
import time
import re 
import pandas as pd
from difflib import SequenceMatcher
import numpy as np
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataSynthesizer')

class ErrorLogFilter:
    """
    Enhanced error log filter that groups similar/repetitive errors and provides
    clean summaries instead of flooding logs with thousands of similar entries.
    """
    
    def __init__(self, similarity_threshold: float = 0.9):
        """
        Initialize the error filter.
        """
        self.similarity_threshold = similarity_threshold
        self.error_groups = {}  # Will store normalized_pattern -> {count, examples, original_errors}
        
    def normalize_error_message(self, error_msg: str) -> str:
        """
        Normalize an error message by removing/standardizing variable parts like:
        - Row numbers (Row 1234, row 1234, in row 1234, etc.)
        - Line numbers 
        - Timestamps
        - Memory addresses
        - File paths with line numbers
        """
        normalized = error_msg
        
        # Remove timestamps at the start of lines
        normalized = re.sub(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,\.]\d{3}', '[TIMESTAMP]', normalized)
        
        # Normalize various row number patterns
        row_patterns = [
            r'\bRow \d+\b',           # "Row 1234"
            r'\brow \d+\b',           # "row 1234" 
            r'\bin row \d+\b',        # "in row 1234"
            r'\bat row \d+\b',        # "at row 1234"
            r'\bfor row \d+\b',       # "for row 1234"
            r'\brow #\d+\b',          # "row #1234"
            r'\bRow #\d+\b',          # "Row #1234"
            r'\bindex \d+\b',         # "index 1234"
            r'\bposition \d+\b',      # "position 1234"
        ]
        
        for pattern in row_patterns:
            normalized = re.sub(pattern, '[ROW_NUM]', normalized, flags=re.IGNORECASE)
        
        # Normalize line numbers in file paths
        normalized = re.sub(r'line \d+', 'line [NUM]', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r':\d+\)', ':[NUM])', normalized)  # For "(file.py:123)"
        
        # Normalize memory addresses
        normalized = re.sub(r'0x[0-9a-fA-F]+', '0x[ADDR]', normalized)
        
        # Normalize specific numeric values that might vary
        normalized = re.sub(r'\b\d+\.\d+\b', '[FLOAT]', normalized)  # Decimal numbers
        normalized = re.sub(r'\b\d{4,}\b', '[LARGE_NUM]', normalized)  # Large integers (4+ digits)
        
        # Clean up extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings using SequenceMatcher."""
        return SequenceMatcher(None, str1, str2).ratio()
    
    def find_similar_group(self, normalized_error: str) -> str:
        """
        Find an existing error group that's similar to the given normalized error.
        Returns the group key if found, None otherwise.
        """
        for existing_pattern in self.error_groups.keys():
            similarity = self.calculate_similarity(normalized_error, existing_pattern)
            if similarity >= self.similarity_threshold:
                return existing_pattern
        return None
    
    def add_error(self, error_message: str):
        """
        Add an error message to the filter for grouping.
        """
        normalized = self.normalize_error_message(error_message)
        
        # Try to find a similar existing group
        similar_group = self.find_similar_group(normalized)
        
        if similar_group:
            # Add to existing group
            self.error_groups[similar_group]['count'] += 1
            # Keep first few examples
            if len(self.error_groups[similar_group]['examples']) < 3:
                self.error_groups[similar_group]['examples'].append(error_message)
        else:
            # Create new group
            self.error_groups[normalized] = {
                'count': 1,
                'examples': [error_message],
                'pattern': normalized
            }
    
    def process_stderr_bulk(self, stderr_content: str) -> List[str]:
        """
        Process a bulk stderr content and return filtered/summarized errors.
        """
        # Split into lines and process each
        lines = stderr_content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and self._is_actual_error_line(line):
                self.add_error(line)
        
        return self.generate_summary()
    
    def _is_actual_error_line(self, line: str) -> bool:
        """
        Determine if a line contains an actual error (not just INFO/DEBUG).
        """
        error_indicators = [
            'error', 'exception', 'failed', 'traceback', 'critical', 'fatal',
            'importerror', 'modulenotfounderror', 'syntaxerror', 'attributeerror',
            'keyerror', 'valueerror', 'typeerror', 'runtimeerror'
        ]
        
        info_patterns = [
            r'- INFO -', r'- DEBUG -', r'- WARNING -'
        ]
        
        line_lower = line.lower()
        
        # Skip if it's clearly just informational
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in info_patterns):
            return False
            
        # Include if it has error indicators
        return any(indicator in line_lower for indicator in error_indicators)
    
    def generate_summary(self) -> List[str]:
        """
        Generate a clean summary of all grouped errors.
        """
        summary_lines = []
        
        # Sort by count (most frequent first)
        sorted_groups = sorted(
            self.error_groups.items(), 
            key=lambda x: x[1]['count'], 
            reverse=True
        )
        
        for pattern, group_info in sorted_groups:
            count = group_info['count']
            examples = group_info['examples']
            
            if count == 1:
                # Single occurrence, just show the original
                summary_lines.append(examples[0])
            else:
                # Multiple occurrences, show summary + example
                summary_lines.append(f"ERROR REPEATED {count} times: {examples[0]}")
                
                # If we have multiple examples showing variation, add them
                if len(examples) > 1:
                    for i, example in enumerate(examples[1:], 2):
                        if i <= 2:  # Show max 2 additional examples
                            summary_lines.append(f"  Example {i}: {example}")
        
        return summary_lines
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the filtered errors.
        """
        total_errors = sum(group['count'] for group in self.error_groups.values())
        unique_patterns = len(self.error_groups)
        
        return {
            'total_error_lines': total_errors,
            'unique_error_patterns': unique_patterns,
            'reduction_ratio': (total_errors - unique_patterns) / max(total_errors, 1),
            'most_common_errors': [
                {
                    'pattern': group['pattern'][:100] + '...' if len(group['pattern']) > 100 else group['pattern'],
                    'count': group['count']
                }
                for _, group in sorted(
                    self.error_groups.items(), 
                    key=lambda x: x[1]['count'], 
                    reverse=True
                )[:5]
            ]
        }

class CodeExecutionAgent:
    """
    Component responsible for executing the generated data synthesis code
    and validating the generated data.
    """

    def __init__(self, timeout_seconds: int = 300):
        """
        Initialize the Code Execution Agent component.
        """
        self.timeout_seconds = timeout_seconds
        logger.info(f"Code Execution Agent initialized with {timeout_seconds}s timeout")

    def execute_code(self, code_path: str) -> Dict[str, Any]:
        """
        Execute the generated Python code.
        """
        logger.info(f"Starting execution of code from {code_path}")

        result = {
            "success": False,
            "error": None,
            "traceback": None, # Will store unique tracebacks or raw stderr
            "output": None,
            "execution_time": 0
        }

        # Validate file existence
        if not os.path.exists(code_path):
            error_msg = f"Code file not found at {code_path}"
            logger.error(error_msg)
            result["error"] = error_msg
            return result

        start_time = time.time()

        try:
            # Execute the code in a subprocess
            process = subprocess.Popen(
                [sys.executable, code_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for process to complete, with a timeout using subprocess.communicate()
            try:
                stdout, stderr = process.communicate(timeout=self.timeout_seconds) 
                exit_code = process.returncode

            except subprocess.TimeoutExpired:
                # This catches the timeout from communicate()
                process.kill() # Terminate the process
                # Get any remaining output after killing the process
                stdout, stderr = process.communicate()
                # Raise a consistent error for reporting
                error_msg = f"Code execution timed out after {self.timeout_seconds} seconds (subprocess timeout)"
                logger.error(error_msg)
                result["error"] = error_msg
                result["traceback"] = "Code execution timed out." # Store simple message for traceback
                return result # Return immediately on timeout


            # Process stderr and exit code
            if exit_code != 0:
                result["success"] = False
                result["output"] = stdout
                result["error"] = f"Execution failed with exit code {exit_code}."
    
                # Only analyze stderr for tracebacks if there's actual error content
                if stderr:
                    unique_tracebacks = self._analyze_stderr_for_unique_errors(stderr)
                    result["traceback"] = unique_tracebacks
                    if unique_tracebacks:
                        logger.error(f"Code execution failed. Found {len(unique_tracebacks)} unique error(s) in stderr.")
                        logger.error("Example Traceback:\n" + unique_tracebacks[0])
                    else:
                        result["traceback"] = [stderr]

            elif stderr and self._contains_actual_errors(stderr):
                # Only treat stderr as failure if it contains actual errors, not just INFO/logging
                result["success"] = False
                result["output"] = stdout
                result["error"] = "Execution produced stderr with actual errors."
                unique_tracebacks = self._analyze_stderr_for_unique_errors(stderr)
                result["traceback"] = unique_tracebacks
                if unique_tracebacks:
                    logger.error(f"Code execution failed. Found {len(unique_tracebacks)} unique error(s) in stderr.")
                    logger.error("Example Traceback:\n" + unique_tracebacks[0])
                else:
                    result["traceback"] = [stderr]

            else:
                # Success case - exit_code == 0 and no actual errors in stderr
                logger.info(f"Code executed successfully in {time.time() - start_time:.2f} seconds")
                result["success"] = True
                result["output"] = stdout
                # Log stderr as informational if present but not error-worthy
                if stderr:
                    logger.info(f"Code execution completed with informational stderr output: {stderr[:200]}...")


        except Exception as e:
            error_msg = f"Unexpected error during execution: {str(e)}"
            logger.error(error_msg)
            result["error"] = error_msg
            result["traceback"] = [traceback.format_exc()] # Store unexpected error traceback as a list

        result["execution_time"] = round(time.time() - start_time, 2) # Ensure execution time is recorded even on error

        return result

    def _analyze_stderr_for_unique_errors(self, stderr_output: str) -> List[str]:
        """
        Enhanced version that uses the ErrorLogFilter for better deduplication.
        """
        if not stderr_output.strip():
            return []
    
        # Use the enhanced filter
        error_filter = ErrorLogFilter(similarity_threshold=0.9)
        filtered_errors = error_filter.process_stderr_bulk(stderr_output)
    
        # Log statistics for debugging
        stats = error_filter.get_statistics()
        if stats['total_error_lines'] > 0:
            logger.info(f"Error filtering stats: {stats['total_error_lines']} total errors "
                    f"reduced to {stats['unique_error_patterns']} unique patterns "
                    f"({stats['reduction_ratio']:.1%} reduction)")
    
        return filtered_errors if filtered_errors else [stderr_output.strip()]

    def validate_generated_data(self, data_path: str, requirements_path: str) -> Dict[str, Any]:
        import pandas as pd

        validation_result = {
            "success": True,
            "summary": "All validation checks passed.",
            "validations": []
        }

        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            return {
                "success": False,
                "summary": f"Failed to load generated data: {e}",
                "validations": []
            }

        # Null percentage checks
        for col in df.columns:
            null_pct = df[col].isna().mean() * 100
            validation_result["validations"].append({
                "name": "null_check",
                "column": col,
                "passed": null_pct < 50.0,
                "details": f"{null_pct:.2f}% null values in column {col}"
            })

        # Uniqueness checks
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            validation_result["validations"].append({
                "name": "uniqueness_check",
                "column": col,
                "passed": unique_ratio > 0.5,
                "details": f"{unique_ratio:.2%} unique values in column {col}"
            })

        # Check against enriched metadata (if available)
        enriched_metadata_path = "pipeline_run_outputs/enriched_metadata.json"
        if os.path.exists(enriched_metadata_path):
            try:
                with open(enriched_metadata_path, "r", encoding="utf-8") as f:
                    enriched_metadata = json.load(f)
                extra_validations = self.validate_against_enriched_metadata(df, enriched_metadata)
                validation_result["validations"].extend(extra_validations)
            except Exception as e:
                validation_result["validations"].append({
                    "name": "enriched_metadata_check",
                    "passed": False,
                    "details": f"Error loading or processing enriched_metadata.json: {e}"
                })

        # Set final status
        failed = [v for v in validation_result["validations"] if not v["passed"]]
        if failed:
            validation_result["success"] = False
            validation_result["summary"] = f"{len(failed)} validation checks failed."

        return validation_result


    def validate_against_enriched_metadata(self, df: pd.DataFrame, enriched_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        tolerance_null_pct = 1.0
        tolerance_value_dist = 1.0
        tolerance_unique_drop = 0.0

        # Flatten enriched metadata columns
        original_columns = []
        for table_info in enriched_metadata.values():
            original_columns.extend(table_info.get("columns", []))

        original_col_names = {col["Column_name"] for col in original_columns}
        generated_col_names = set(df.columns)

        # Column count check
        if generated_col_names != original_col_names:
            missing = original_col_names - generated_col_names
            extra = generated_col_names - original_col_names
            results.append({
                "name": "column_match_check",
                "passed": False,
                "details": f"Missing columns: {missing}. Unexpected columns: {extra}."
            })

        for orig_col in original_columns:
            col = orig_col["Column_name"]
            if col not in df.columns:
                continue  # already marked above

            series = df[col]
            stat = {}

            # Null percentage
            gen_null_pct = series.isna().mean() * 100
            orig_null_pct = orig_col.get("null_percentage", 0.0)
            delta = abs(gen_null_pct - orig_null_pct)
            stat["passed"] = delta <= tolerance_null_pct
            stat["name"] = "null_percentage_check"
            stat["column"] = col
            stat["details"] = f"Column {col} has {gen_null_pct:.2f}% nulls vs {orig_null_pct:.2f}% (allowed ±{tolerance_null_pct}%)"
            results.append(stat)

            # Unique count check
            gen_unique = series.nunique()
            orig_unique = orig_col.get("num_unique", gen_unique)
            unique_passed = gen_unique >= orig_unique * (1 - tolerance_unique_drop)
            results.append({
                "name": "unique_count_check",
                "column": col,
                "passed": unique_passed,
                "details": f"Column {col} has {gen_unique} unique vs original {orig_unique} (≥{(1 - tolerance_unique_drop)*100:.0f}%)"
            })

            # Min/max range
            try:
                gen_min, gen_max = pd.to_numeric(series, errors='coerce').min(), pd.to_numeric(series, errors='coerce').max()
                orig_min, orig_max = orig_col.get("min_val"), orig_col.get("max_val")
                if orig_min is not None and orig_max is not None:
                    range_pass = orig_min <= gen_min <= gen_max <= orig_max
                    results.append({
                        "name": "range_check",
                        "column": col,
                        "passed": range_pass,
                        "details": f"Column {col} range [{gen_min}, {gen_max}] vs [{orig_min}, {orig_max}]"
                    })
            except Exception:
                pass

            # Categorical value distribution (optional)
            if orig_col.get("data_type") == "categorical" and "value_distribution" in orig_col:
                gen_dist = series.value_counts(normalize=True) * 100
                gen_dist = gen_dist.to_dict()
                for category, expected_pct in orig_col["value_distribution"].items():
                    actual_pct = gen_dist.get(category, 0.0)
                    drift = abs(actual_pct - expected_pct)
                    results.append({
                        "name": "value_distribution_check",
                        "column": col,
                        "passed": drift <= tolerance_value_dist,
                        "details": f"{category} is {actual_pct:.2f}% vs expected {expected_pct:.2f}% (±{tolerance_value_dist}%)"
                    })

        return results

    def run_execution_and_validation(self,
                                code_path: str,
                                requirements_path: str,
                                expected_output_path: str = None) -> Dict[str, Any]:
        """
        Run the full execution and validation process.
        Enhanced to better locate output files.
        """
        logger.info("Starting execution and validation process")

        # Initialize the results structure
        results = {
        "execution": None,
        "validation": None,
        "overall_success": False,
        "summary": ""
        }

        # Execute the code
        execution_result = self.execute_code(code_path)
        results["execution"] = execution_result

        # If execution failed (based on exit code OR stderr), no need to proceed to validation
        if not execution_result["success"]:
            results["summary"] = f"Code execution failed: {execution_result.get('error', 'Unknown error')}"
            logger.error(results["summary"])
            return results

        # Determine output file path if not provided
        output_path = expected_output_path
        if not output_path:
            output_path = "pipeline_run_outputs/synthesized_output_data.csv"  # Default output path

        # If still no output path found, try default/common locations
        if not output_path:
            code_dir = os.path.dirname(code_path)
            current_dir = os.getcwd()
        
            # Common default filenames to check
            default_filenames = [
            "synthesized_output_data.csv",
            "synthetic_fct_ent_casa.csv",
            "generated_data.csv",
            "output.csv",
            "synthesized_output_data.json",
            "output.json"
            ]
        
            # Locations to check
            search_dirs = [
            current_dir,
            code_dir,
            os.path.join(current_dir, "pipeline_run_outputs"),
            os.path.join(code_dir, "pipeline_run_outputs"),
            os.path.join(current_dir, "outputs"),
            os.path.join(code_dir, "outputs"),
            ]
        
            logger.info(f"Searching for output file in directories: {search_dirs}")
        
            for search_dir in search_dirs:
                if not os.path.exists(search_dir):
                    continue
                
                # Check for default filenames
                for filename in default_filenames:
                    potential_path = os.path.join(search_dir, filename)
                    if os.path.exists(potential_path):
                        output_path = potential_path
                        logger.info(f"Found output file at: {output_path}")
                        break
            
                if output_path:
                    break
            
                # Also check for any CSV/JSON files in the directory
                try:
                    for file in os.listdir(search_dir):
                        if file.lower().endswith(('.csv', '.json')) and not file.startswith('.'):
                            potential_path = os.path.join(search_dir, file)
                            # Check if file was recently created (within last few minutes)
                            if os.path.getmtime(potential_path) > (time.time() - 300):  # 5 minutes
                                output_path = potential_path
                                logger.info(f"Found recently created output file at: {output_path}")
                                break
                    if output_path:
                        break
                except OSError:
                    continue

        if not output_path or not os.path.exists(output_path):
            # List files in likely directories for debugging
            debug_info = []
            for debug_dir in [os.getcwd(), os.path.dirname(code_path), "pipeline_run_outputs"]:
                if os.path.exists(debug_dir):
                    try:
                        files = [f for f in os.listdir(debug_dir) if f.lower().endswith(('.csv', '.json'))]
                        debug_info.append(f"{debug_dir}: {files}")
                    except OSError:
                        pass
        
            results["summary"] = f"Code executed successfully but no output file was found. Searched locations: {debug_info}"
            logger.error(results["summary"])
            results["validation"] = {
            "success": False,
            "validations": [],
            "summary": "No output file found for validation"
            }
            return results
        else:
            logger.info(f"Found output file for validation at: {output_path}")
            results["validation"] = {"data_path_validated": output_path}

        validation_result = self.validate_generated_data(output_path, requirements_path)
        results["validation"].update(validation_result)

        results["overall_success"] = execution_result["success"] and results["validation"]["success"]

        if results["overall_success"]:
            results["summary"] = f"Execution and validation completed successfully. {results['validation']['summary']}"
        else:
            results["summary"] = f"Execution succeeded but validation failed. {results['validation']['summary']}"

        logger.info(results["summary"])
        return results

    def save_results(self, results: dict[str, Any], output_path: str = "pipeline_run_outputs/code_execution_report.json") -> str:
        """
        Save the execution and validation results to a file,
        sanitizing non-JSON-serializable types.
        """
        self.output_path = output_path # Ensure output_path is set correctly within the instance
        logger.info(f"Saving execution and validation results to {self.output_path}")

        def sanitize_for_json(obj):
            """Recursively converts non-JSON-serializable types to standard Python types."""
            if isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_for_json(elem) for elem in obj]
            elif isinstance(obj, (bool, np.bool_)): 
                return bool(obj) 
            elif isinstance(obj, (int, np.integer)): 
                return int(obj)
            elif isinstance(obj, (float, np.floating)): 
                return float(obj)
            elif isinstance(obj, datetime.datetime): 
                return obj.isoformat()
            elif isinstance(obj, (set, tuple)): 
                return [sanitize_for_json(elem) for elem in obj]
            elif pd.isna(obj): 
                return None
            return obj

        try:
            sanitized_results = sanitize_for_json(results)
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(sanitized_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results successfully saved to {self.output_path}")
            return self.output_path
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

    def _contains_actual_errors(self, stderr_output: str) -> bool:
        """
        Determines if stderr output contains actual errors vs just informational logging.
        """
        # Look for actual error indicators
        error_indicators = [
            "Traceback (most recent call last):",
            "Error:",
            "Exception:",
            "CRITICAL",
            "FATAL",
            "Failed",
            "ImportError:",
            "ModuleNotFoundError:",
            "SyntaxError:",
            "AttributeError:",
            "KeyError:",
            "ValueError:",
            "TypeError:"
        ]
    
        # Look for non-error log levels that should be ignored
        info_only_patterns = [
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - \w+ - INFO -",
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - \w+ - DEBUG -",
            r"- INFO -",
            r"- DEBUG -"
        ]
    
        # Check if stderr contains actual error indicators
        stderr_lower = stderr_output.lower()
        has_errors = any(indicator.lower() in stderr_lower for indicator in error_indicators)
    
        if has_errors:
            return True
    
        lines = stderr_output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            is_info_only = any(re.search(pattern, line) for pattern in info_only_patterns)
            if not is_info_only:
                return True
    
        return False

if __name__ == "__main__":
    agent = CodeExecutionAgent(timeout_seconds=360)  # 6-minute timeout

    results = agent.run_execution_and_validation(
        code_path="pipeline_run_outputs/generated_data_script.py",
        requirements_path="pipeline_run_outputs/generation_requirements.txt"
    )

    output_path = agent.save_results(results)

    print(f"Execution and validation complete. Results saved to {output_path}")
    print(f"Summary: {results['summary']}")