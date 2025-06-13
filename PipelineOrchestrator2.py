import json
import os
import logging
import shutil
import re
from typing import Dict, Any, Optional, Tuple, List

from InputProcessor import InputProcessor           #type: ignore
from LLMAnalyzer import LLMAnalyzer                 #type: ignore
from WebSearchAgent import WebSearchAgent           #type: ignore
from CodeGenerator import CodeGenerator             #type: ignore
from CodeReviewer import CodeReviewer               #type: ignore
from CodeExecutionAgent import CodeExecutionAgent   #type: ignore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() 
    ]
)
logger = logging.getLogger('PipelineOrchestrator')

class PipelineOrchestrator:
    """
    Orchestrates the LLM-Powered Data Synthesizer pipeline, managing the flow
    between different components, handling feedback loops, and producing
    the final synthetic data.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the PipelineOrchestrator with necessary configurations
        and all agent components.
        """
        self.config = config
        self.api_key = config["google_api_key"]

        self.output_dir = config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory set to: {os.path.abspath(self.output_dir)}")

        self.input_processor = InputProcessor()
        self.llm_analyzer = LLMAnalyzer(api_key=self.api_key)
        self.web_search_agent = WebSearchAgent(
            api_key=self.api_key,
            cx=config.get("google_search_engine_id")
        )
        self.code_generator = CodeGenerator(api_key=self.api_key)
        self.code_reviewer = CodeReviewer(api_key=self.api_key)
        self.code_execution_agent = CodeExecutionAgent(
            timeout_seconds=config["code_execution_timeout"]
        )

        # Paths for intermediate files - constructed using output_dir
        self.enriched_metadata_path = os.path.join(self.output_dir, config["enriched_metadata_filename"])
        self.requirements_path = os.path.join(self.output_dir, config["requirements_doc_filename"])
        self.ordered_columns_path = os.path.join(self.output_dir, config["ordered_columns_filename"])
        self.search_results_path = os.path.join(self.output_dir, config["search_results_filename"])
        self.generated_code_path = os.path.join(self.output_dir, config["generated_code_filename"])
        self.review_results_path = os.path.join(self.output_dir, config["review_results_filename"])
        self.execution_results_path = os.path.join(self.output_dir, config["execution_results_filename"])
        self.original_column_order_path = os.path.join(self.output_dir, config["original_column_order_filename"])
        
        # Variables to store intermediate results for the feedback loop
        self.enriched_metadata: Optional[Dict] = None
        self.column_relationships: Optional[Dict] = None
        self.ordered_columns: Optional[list] = None
        self.original_column_order: Optional[List[str]] = None
        self.current_feedback: Optional[str] = None
        self.last_review_results: Optional[Dict] = None # for feedback loop
        self.last_execution_results: Optional[Dict] = None # for feedback loop
        self.final_synthetic_data_path: Optional[str] = None

        # Stores the generated code for each logical section of the script
        self.code_blocks: Dict[str, str] = {}
        self.frozen_sections: Dict[str, bool] = {}
        self.last_attempted_code_blocks: Dict[str, str] = {}

    def _assemble_full_script(self) -> str:
        """
        Assembles the full Python script from individual code blocks.
        Order of assembly is critical for a functional script.
        """
        assembled_code_parts: List[str] = []

        section_order_template = [
            "imports_setup",
            "all_column_logic",
            "main_orchestration",
            "file_output"
        ]

        for section_name in section_order_template:
            if section_name in self.code_blocks:
                assembled_code_parts.append(self.code_blocks[section_name])
            else:
                logger.warning(f"Missing expected code block for section: {section_name}")

        full_script = "\n\n".join(assembled_code_parts)

        try:
            with open(self.generated_code_path, 'w', encoding='utf-8') as f:
                f.write(full_script)
            logger.info(f"Full script assembled and saved to {self.generated_code_path}")
        except Exception as e:
            logger.error(f"Error assembling and saving full script: {e}")
            raise

        return full_script

    def _load_existing_artifacts(self):
        """
        Load existing artifacts needed for step 4 onwards when starting from a later step.
        """
        logger.info("Loading existing artifacts for starting from step 4...")
        try:
            if not os.path.exists(self.enriched_metadata_path):
                logger.error(f"Missing enriched metadata file at {self.enriched_metadata_path}")
                return False
            
            if not os.path.exists(self.requirements_path):
                logger.error(f"Missing requirements document at {self.requirements_path}")
                return False
            
            if not os.path.exists(self.ordered_columns_path):
                logger.error(f"Missing ordered columns file at {self.ordered_columns_path}")
                return False
            
            if not os.path.exists(self.search_results_path):
                logger.warning(f"Search results file not found at {self.search_results_path}. Creating an empty one.")
                with open(self.search_results_path, 'w', encoding='utf-8') as f:
                    json.dump({}, f)
                
            if not os.path.exists(self.original_column_order_path):
                logger.warning(f"Original column order file not found at {self.original_column_order_path}. This may cause issues.")
            else:
                with open(self.original_column_order_path, 'r', encoding='utf-8') as f:
                    self.original_column_order = json.load(f)
                    logger.info("Original column order loaded successfully.")
        
            with open(self.ordered_columns_path, 'r', encoding='utf-8') as f:
                self.ordered_columns = json.load(f)
                logger.info("Ordered columns loaded successfully.")
            
            with open(self.enriched_metadata_path, 'r', encoding='utf-8') as f:
                self.enriched_metadata = json.load(f)
                logger.info("Enriched metadata loaded successfully.")
            
            # Check if there's a previous execution or review results to use for feedback
            if os.path.exists(self.review_results_path):
                try:
                    with open(self.review_results_path, 'r', encoding='utf-8') as f:
                        self.last_review_results = json.load(f)
                    logger.info("Previous review results loaded for feedback.")
                except Exception as e:
                    logger.warning(f"Could not load previous review results: {e}")
                
            if os.path.exists(self.execution_results_path):
                try:
                    with open(self.execution_results_path, 'r', encoding='utf-8') as f:
                        self.last_execution_results = json.load(f)
                    logger.info("Previous execution results loaded for feedback.")
                except Exception as e:
                    logger.warning(f"Could not load previous execution results: {e}")

            if getattr(self, "start_from_step_4", False):
                logger.info("start_from_step_4 is set — skipping stale review and execution results.")
                self.last_review_results = None
                self.last_execution_results = None
                self.code_blocks = {}
                self.frozen_sections = {}
                self.last_attempted_code_blocks = {}
            
            return True
        except Exception as e:
            logger.error(f"Failed to load existing artifacts: {e}", exc_info=True)
            return False
        
    def _get_section_specific_feedback(self, section_tag: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extracts feedback and a list of all critical structured issues for a section.
        Returns a tuple of (full_feedback_string, list_of_structured_critical_issues).
        """
        feedback_parts = []
        all_issues = []
        structured_critical_issues = [] 

        # (Existing helper function _summarize_traceback remains the same)
        def _summarize_traceback(traceback_text: str) -> Tuple[str, Optional[int]]:
            if not traceback_text:
                return "No traceback available.", None
            if isinstance(traceback_text, list):
                traceback_text = "\n".join(traceback_text)
            line_match = re.search(r'File ".*generated_data_script.py", line (\d+)', traceback_text)
            line_number = int(line_match.group(1)) if line_match else None
            lines = [line for line in traceback_text.strip().split('\n') if line.strip()]
            error_message = lines[-1] if lines else "Traceback found, but could not extract error message."
            return error_message, line_number

        # 1. Gather all issues
        if self.last_review_results:
            for issue in self.last_review_results.get("critical_issues", []):
                if issue.get("section_tag") == section_tag:
                    all_issues.append({"priority": 1, "type": "critical_review", "data": issue})
        if self.last_execution_results:
            exec_results_safe = self.last_execution_results if self.last_execution_results is not None else {}
            exec_data = exec_results_safe.get("execution", {})
            if not exec_data.get("success", True) and exec_data.get("traceback"):
                all_issues.append({"priority": 2, "type": "execution_error", "data": exec_data})
            if section_tag == "all_column_logic":
                validation_data = exec_results_safe.get("validation", {})
                if isinstance(validation_data, dict):
                    failed_vals = [v for v in validation_data.get("validations", []) if not v.get("passed", True)]
                    for val in failed_vals:
                        all_issues.append({"priority": 4, "type": "validation_failure", "data": val})

        if not all_issues:
            return "", []

        # 2. Filter for high-priority issues and create structured JSON for each
        high_priority_issues = [issue for issue in all_issues if issue["priority"] in [1, 2, 4]]

        for issue in high_priority_issues:
            structured_issue = None
            if issue["type"] == "critical_review":
                issue_data = issue["data"]
                structured_issue = {
                    "issue_type": issue_data.get("type"), "location": issue_data.get("location"),
                    "message": issue_data.get("description"), "recommendation": issue_data.get("recommendation"),
                    "section_tag": section_tag
                }

            elif issue["type"] == "execution_error":
                exec_data = issue["data"]
                error_msg, line_num = _summarize_traceback(exec_data.get("traceback", ""))
                location = f"Line ~{line_num}" if line_num else "Unknown location in script"
                structured_issue = {
                    "issue_type": "execution_error", "location": location,
                    "message": f"Primary error from traceback: {error_msg}",
                    "recommendation": "Analyze the traceback to fix the runtime error.", "section_tag": section_tag
                }

            elif issue["type"] == "validation_failure":
                val_data = issue["data"]
                msg = f"Data Validation Failed for `{val_data.get('column')}`: `{val_data.get('name')}` because `{val_data.get('details')}`."
                structured_issue = {
                    "issue_type": "validation_failure", "location": f"Function related to column '{val_data.get('column')}'",
                    "message": msg, "recommendation": f"Review the `generate_{val_data.get('column')}` function (or relevant part of column logic).",
                    "section_tag": section_tag
                }

            if structured_issue and structured_issue not in structured_critical_issues:
                 structured_critical_issues.append(structured_issue)

        # 3. Build the natural language feedback string from all issues found for the section
        feedback_parts.append("Please analyze the following feedback specific to this code section:")
        if self.last_review_results:
            crit_issues = [i["data"] for i in all_issues if i["type"] == "critical_review"]
            non_crit_issues = [i["data"] for i in all_issues if i["type"] == "non_critical_review"]
            if crit_issues:
                feedback_parts.append("\n--- Critical Code Review Feedback ---")
                for issue in crit_issues:
                    feedback_parts.append(f"- {issue.get('description', '')} → {issue.get('recommendation', '')}")
            if non_crit_issues:
                 feedback_parts.append("\n--- Non-Critical Code Review Feedback ---")
                 for issue in non_crit_issues:
                    feedback_parts.append(f"- {issue.get('description', '')} → {issue.get('recommendation', '')}")

        # Add execution and validation feedback
        if self.last_execution_results:
            if any(i["type"] == "execution_error" for i in all_issues):
                error_msg, line_num = _summarize_traceback(self.last_execution_results.get("execution", {}).get("traceback", ""))
                feedback_parts.append("\n--- Execution Error Feedback ---")
                feedback_parts.append(f"Execution failed. Primary error from traceback: {error_msg} at line ~{line_num}.")
            
            val_failures = [i["data"] for i in all_issues if i["type"] == "validation_failure"]
            if val_failures:
                feedback_parts.append("\n--- Data Validation Feedback ---")
                for val in val_failures:
                    feedback_parts.append(f"- Data Validation Failed for `{val.get('column')}`: `{val.get('name')}` because `{val.get('details')}`. Review the `generate_{val.get('column')}` function.")

        return "\n".join(feedback_parts), structured_critical_issues

    def _step_1_process_inputs(self) -> bool:
        """
        Loads and processes initial metadata, relationships, and sample data.
        """
        logger.info("Starting Step 1: Input Processing...")
        try:
            enriched_metadata_dict, column_relationships_dict, _,original_column_order_list = \
                self.input_processor.process_all_inputs(
                    metadata_filepath=self.config["input_metadata_filepath"],
                    relationships_filepath=self.config["input_relationships_filepath"],
                    samples_filepath=self.config["input_samples_filepath"],
                )

            with open(self.enriched_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(enriched_metadata_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"Enriched metadata saved to {self.enriched_metadata_path}")

            with open(self.original_column_order_path, 'w', encoding='utf-8') as f: # <-- Save original order
                json.dump(original_column_order_list, f, ensure_ascii=False, indent=2)
            logger.info(f"Original column order saved to {self.original_column_order_path}")

            self.enriched_metadata = enriched_metadata_dict
            self.column_relationships = column_relationships_dict
            self.original_column_order = original_column_order_list
            logger.info("Step 1: Input Processing COMPLETED.")
            return True
        except Exception as e:
            logger.error(f"Step 1: Input Processing FAILED. Error: {e}", exc_info=True)
            return False

    def _step_2_analyze_metadata(self) -> bool:
        """
        Analyzes the enriched metadata to determine generation strategies and order.
        """
        logger.info("Starting Step 2: Metadata Analysis (LLM)...")
        if not self.enriched_metadata or not self.column_relationships:
            logger.error("Step 2: Cannot proceed without enriched metadata and column relationships from Step 1.")
            return False
        try:
            ordered_columns_list, requirements_doc_str = \
                self.llm_analyzer.analyze_metadata(
                    enriched_metadata=self.enriched_metadata,
                    column_relationships=self.column_relationships
                )

            with open(self.requirements_path, 'w', encoding='utf-8') as f:
                f.write(requirements_doc_str)
            logger.info(f"Requirements document saved to {self.requirements_path}")

            with open(self.ordered_columns_path, 'w', encoding='utf-8') as f:
                json.dump(ordered_columns_list, f, ensure_ascii=False, indent=2)
            logger.info(f"Ordered columns list saved to {self.ordered_columns_path}")

            self.ordered_columns = ordered_columns_list
            logger.info("Step 2: Metadata Analysis COMPLETED.")
            return True
        except Exception as e:
            logger.error(f"Step 2: Metadata Analysis FAILED. Error: {e}", exc_info=True)
            return False

    def _step_3_web_search_if_needed(self) -> bool:
        """
        Performs web searches if the LLMAnalyzer indicated a need for any columns.
        """
        logger.info("Starting Step 3: Web Search (Optional)...")
        if not self.ordered_columns:
            logger.error("Step 3: Cannot proceed without ordered columns from Step 2.")
            return False

        # Check if any column requires web search
        search_needed = any(
            col.get("generation_strategy") == "library_search" and col.get("search_queries")
            for col in self.ordered_columns
        )

        if not search_needed:
            logger.info("Step 3: No web search required based on LLM analysis. SKIPPING.")
            if not os.path.exists(self.search_results_path):
                 with open(self.search_results_path, 'w', encoding='utf-8') as f:
                    json.dump({}, f)
                 logger.info(f"Created empty search results file at {self.search_results_path}")
            return True

        try:
            default_search_results_file = "search_results.json"
            if os.path.exists(default_search_results_file):
                 os.remove(default_search_results_file) # Clean up 

            self.web_search_agent.process_batch_enhanced_searches(ordered_columns=self.ordered_columns)

            if os.path.exists(default_search_results_file):
                shutil.move(default_search_results_file, self.search_results_path)
                logger.info(f"Search results saved to {self.search_results_path}")
            elif os.path.exists(self.search_results_path):
                 logger.info(f"Search results already at {self.search_results_path} (possibly from direct save by agent)")
            else:
                logger.warning(f"WebSearchAgent did not produce {default_search_results_file}. Assuming no results or error.")
                with open(self.search_results_path, 'w', encoding='utf-8') as f:
                    json.dump({}, f)

            logger.info("Step 3: Web Search COMPLETED.")
            return True
        except Exception as e:
            logger.error(f"Step 3: Web Search FAILED. Error: {e}", exc_info=True)
            with open(self.search_results_path, 'w', encoding='utf-8') as f:
                json.dump({}, f)
            return False 

    def _step_4_generate_code(self, attempt: int) -> bool:
        """
        Generates Python code for data synthesis in sections using LLM.
        Incorporates feedback and previous code for iterative improvement.
        """
        logger.info(f"Starting Step 4: Code Generation (Attempt {attempt})...")
        MAX_FEEDBACK_LEN = 5000
        if not self.requirements_path or not self.ordered_columns_path:
            logger.error("Step 4: Missing requirements or ordered columns path.")
            return False

        try:
            with open(self.requirements_path, 'r', encoding='utf-8') as f:
                requirements_content = f.read()
            with open(self.search_results_path, 'r', encoding='utf-8') as f:
                search_results_data = json.load(f)

            sections_to_regenerate: List[str] = []
            all_sections = ["imports_setup", "all_column_logic", "main_orchestration", "file_output"]

            if attempt == 1:
                sections_to_regenerate = all_sections.copy()
            else:
                sections_to_regenerate = [
                section for section in all_sections 
                if not self.frozen_sections.get(section, False)
            ]
                
            logger.info(f"*****Sections to regenerate in this attempt: {sections_to_regenerate}")

            # --- Generate Imports and Setup Block ---
            imports_setup_section_name = "imports_setup"
            if imports_setup_section_name in sections_to_regenerate:
                logger.info(f"Generating '{imports_setup_section_name}' block...")
                feedback_string, structured_issues_list = self._get_section_specific_feedback(imports_setup_section_name)
                if len(feedback_string) > MAX_FEEDBACK_LEN:
                    feedback_string = f"...(feedback truncated)...\n{feedback_string[-MAX_FEEDBACK_LEN:]}"
                    print("Feedback truncated")
                
                generated_block = self.code_generator.generate_imports_setup_block(
                    general_instructions={
                        "python_version": "3.10+", 
                        "output_filename": os.path.join(self.config["output_dir"], f"{self.config['final_data_filename_prefix']}"),
                        "output_format": self.config["output_format"],
                        "num_rows": self.config["num_rows"]
                    },
                    search_results=search_results_data,
                    relevant_ordered_columns_data=requirements_content,
                    feedback=feedback_string, 
                    previous_code=self.last_attempted_code_blocks.get(imports_setup_section_name),
                    structured_critical_issues=structured_issues_list 
                )
                self.code_blocks[imports_setup_section_name] = generated_block
                self.last_attempted_code_blocks[imports_setup_section_name] = generated_block
            elif imports_setup_section_name not in self.code_blocks:
                logger.warning(f"'{imports_setup_section_name}' not in sections to regenerate and not previously generated. This might lead to incomplete script.")
                return False # Or attempt to generate it without feedback

            # --- Generate Column Logic Blocks ---
            all_column_logic_section_name = "all_column_logic"
            if all_column_logic_section_name in sections_to_regenerate:
                logger.info(f"Generating '{all_column_logic_section_name}' block...")
                feedback_string, structured_issues_list = self._get_section_specific_feedback(all_column_logic_section_name)
                if len(feedback_string) > MAX_FEEDBACK_LEN:
                    feedback_string = f"...(feedback truncated)...\n{feedback_string[-MAX_FEEDBACK_LEN:]}"
                    print("Feedback truncated")
                
                generated_block = self.code_generator.generate_all_column_logic_block(
                    all_column_requirements=requirements_content,
                    search_results=search_results_data,
                    imports_block_content=self.code_blocks.get(imports_setup_section_name, ""),
                    feedback=feedback_string,
                    previous_code=self.last_attempted_code_blocks.get(all_column_logic_section_name),
                    structured_critical_issues=structured_issues_list
                )
                self.code_blocks[all_column_logic_section_name] = generated_block
                self.last_attempted_code_blocks[all_column_logic_section_name] = generated_block
            elif all_column_logic_section_name not in self.code_blocks:
                logger.warning(f"'{all_column_logic_section_name}' not in sections to regenerate and not previously generated. This might lead to incomplete script.")
                return False

            # --- Generate Main Orchestration Logic Block ---
            main_orchestration_section_name = "main_orchestration"
            if main_orchestration_section_name in sections_to_regenerate:
                logger.info(f"Generating '{main_orchestration_section_name}' block...")
                feedback_string, structured_issues_list = self._get_section_specific_feedback(main_orchestration_section_name)
                if len(feedback_string) > MAX_FEEDBACK_LEN:
                    feedback_string = f"...(feedback truncated)...\n{feedback_string[-MAX_FEEDBACK_LEN:]}"
                    print("Feedback truncated")

                generated_block = self.code_generator.generate_main_orchestration_block(
                    imports_block_content=self.code_blocks.get(imports_setup_section_name, ""),
                    all_column_logic_code=self.code_blocks.get(all_column_logic_section_name, ""),
                    data_structure_instructions={"type": "DataFrame", "name": "df"}, 
                    original_column_order=self.original_column_order,
                    requirements_content=requirements_content,
                    feedback=feedback_string,
                    previous_code=self.last_attempted_code_blocks.get(main_orchestration_section_name),
                    structured_critical_issues=structured_issues_list 
                )
                self.code_blocks[main_orchestration_section_name] = generated_block
                self.last_attempted_code_blocks[main_orchestration_section_name] = generated_block
            elif main_orchestration_section_name not in self.code_blocks:
                logger.warning(f"'{main_orchestration_section_name}' not in sections to regenerate and not previously generated. This might lead to incomplete script.")
                return False

            # --- Generate File Output Logic Block ---
            file_output_section_name = "file_output"
            if file_output_section_name in sections_to_regenerate:
                logger.info(f"Generating '{file_output_section_name}' block...")
                feedback_string, structured_issues_list = self._get_section_specific_feedback(file_output_section_name)
                if len(feedback_string) > MAX_FEEDBACK_LEN:
                    feedback_string = f"...(feedback truncated)...\n{feedback_string[-MAX_FEEDBACK_LEN:]}"
                    print("Feedback truncated")

                generated_block = self.code_generator.generate_file_output_block(
                    imports_block_content=self.code_blocks.get(imports_setup_section_name, ""),
                    all_column_logic_code=self.code_blocks.get(all_column_logic_section_name, ""),
                    main_orchestration_dataframe_variable="df", 
                    original_column_order=self.original_column_order,
                    output_format=self.config["output_format"],
                    feedback=feedback_string,
                    previous_code=self.last_attempted_code_blocks.get(file_output_section_name),
                    structured_critical_issues=structured_issues_list 
                )
                self.code_blocks[file_output_section_name] = generated_block
                self.last_attempted_code_blocks[file_output_section_name] = generated_block
            elif file_output_section_name not in self.code_blocks:
                logger.warning(f"'{file_output_section_name}' not in sections to regenerate and not previously generated. This might lead to incomplete script.")
                return False

            # After all necessary sections are generated/updated, assemble the full script
            self._assemble_full_script()

            logger.info(f"Step 4: Code Generation (Attempt {attempt}) COMPLETED.")
            return True
        except Exception as e:
            logger.error(f"Step 4: Code Generation (Attempt {attempt}) FAILED. Error: {e}", exc_info=True)
            return False

    def _step_5_review_code(self, attempt: int) -> Tuple[bool, Optional[Dict]]:
        """
        Performs static analysis/review of the generated code.
        """
        logger.info(f"Starting Step 5: Code Review (Attempt {attempt})...")
        if not os.path.exists(self.generated_code_path) or not os.path.exists(self.requirements_path):
            logger.error("Step 5: Missing generated code or requirements path for review.")
            return False, None

        try:
            review_results = self.code_reviewer.review_code(
                code_path=self.generated_code_path,
                requirements_path=self.requirements_path
            )
            self.code_reviewer.save_review_results(review_results, self.review_results_path)
            logger.info(f"Code review results saved to {self.review_results_path}")

            # Define "acceptable" review
            is_acceptable = review_results.get("pass", False) and \
                            not review_results.get("critical_issues")

            if is_acceptable:
                logger.info(f"Step 5: Code Review (Attempt {attempt}) PASSED.")
            else:
                logger.warning(f"Step 5: Code Review (Attempt {attempt}) FAILED or found critical issues.")
                summary = review_results.get("summary", "No summary.")
                critical_issues = review_results.get("critical_issues", [])
                logger.warning(f"Review Summary: {summary}")
                if critical_issues:
                    logger.warning("Critical Issues Found:")
                    for issue in critical_issues:
                        logger.warning(f"  - {issue.get('type')}: {issue.get('description')} (Location: {issue.get('location')})")

            return is_acceptable, review_results
        except Exception as e:
            logger.error(f"Step 5: Code Review (Attempt {attempt}) FAILED. Error: {e}", exc_info=True)
            return False, None

    def _step_6_execute_code(self, attempt: int) -> Tuple[bool, Optional[Dict]]:
        """
        Executes the generated (and reviewed) code and validates its output.
        """
        logger.info(f"Starting Step 6: Code Execution & Validation (Attempt {attempt})...")
        if not os.path.exists(self.generated_code_path) or not os.path.exists(self.requirements_path):
            logger.error("Step 6: Missing generated code or requirements path for execution.")
            return False, None

        try:
            execution_results = self.code_execution_agent.run_execution_and_validation(
                code_path=self.generated_code_path,
                requirements_path=self.requirements_path,
            )
            if execution_results is None:
                logger.warning("execution_results is unexpectedly None. Inserting fallback.")
                execution_results = {
            "execution": {"success": False, "traceback": "Unavailable"},
            "validation": {"success": False, "summary": "Unavailable"},
            "summary": "Execution failed but returned None."
            }
            self.code_execution_agent.save_results(execution_results, self.execution_results_path)
            logger.info(f"Execution and validation results saved to {self.execution_results_path}")

            is_successful = execution_results.get("overall_success", False)

            if is_successful:
                logger.info(f"Step 6: Code Execution & Validation (Attempt {attempt}) SUCCEEDED.")
                if execution_results.get("validation") and execution_results["validation"].get("data_path_validated"):
                    self.final_synthetic_data_path = execution_results["validation"]["data_path_validated"]
                else: 
                    potential_path = os.path.join(self.output_dir, f"{self.config['final_data_filename_prefix']}.{self.config['output_format']}")
                    print(potential_path)
                    if os.path.exists(potential_path):
                         self.final_synthetic_data_path = potential_path


            else:
                logger.warning(f"Step 6: Code Execution & Validation (Attempt {attempt}) FAILED.")
                summary = execution_results.get("summary", "No summary.")
                logger.warning(f"Execution/Validation Summary: {summary}")
                if execution_results.get("execution") and execution_results["execution"].get("error"):
                    logger.warning(f"Execution Error: {execution_results['execution']['error']}")
                    logger.warning(f"Traceback: {execution_results['execution'].get('traceback', 'N/A')}")
                if execution_results.get("validation") and not execution_results["validation"].get("success"):
                    logger.warning(f"Validation Failures: {execution_results['validation'].get('summary', 'N/A')}")


            return is_successful, execution_results
        except Exception as e:
            logger.error(f"Step 6: Code Execution & Validation (Attempt {attempt}) FAILED. Error: {e}", exc_info=True)
            return False, None

    def _gather_feedback(self, review_results: Dict[str, Any], execution_results: Dict[str, Any]) -> str:
        """
        Gathers feedback, updates frozen_sections based on review and execution.
        """
        self.last_review_results = review_results
        self.last_execution_results = execution_results

        # Handle None execution_results
        if execution_results is None:
            logger.error("execution_results is None - creating fallback execution results")
            execution_results = {
                "execution": {"success": False, "traceback": "Execution results unavailable"},
                "validation": {"success": False, "summary": "Validation results unavailable"},
                "summary": "Execution failed and returned None"
            }
            self.last_execution_results = execution_results

        all_sections = ["imports_setup", "all_column_logic", "main_orchestration", "file_output"]
        sections_with_issues = set()

        # 1. Add sections with review issues
        if review_results:  # Also check if review_results is not None
            for issue in review_results.get("critical_issues", []): #+ review_results.get("non_critical_issues", []):
                if issue.get("section_tag"):
                    sections_with_issues.add(issue.get("section_tag"))

        # 2. Add sections with execution/validation issues
        execution_failed = not execution_results.get("execution", {}).get("success", True)
        validation_data = execution_results.get("validation")
        if validation_data is None:
            validation_failed = True  
        else:
            validation_failed = not validation_data.get("success", True)

        if execution_failed:
            traceback_related_sections = set()
            if review_results:  # Check if review_results is not None
                traceback_related_sections = {
                    issue.get("section_tag") for issue in review_results.get("critical_issues", [])
                    if issue.get("traceback") and issue.get("section_tag")
                }
            if traceback_related_sections:
                logger.info(f"Execution failure linked by reviewer to sections: {traceback_related_sections}")
                sections_with_issues.update(traceback_related_sections)
            else:
                logger.warning("Execution failure occurred, but reviewer did not link it to a specific section. Unfreezing all sections as a precaution.")
                sections_with_issues.update(all_sections)
    
        if validation_failed:
            sections_with_issues.add("all_column_logic")

        # 3. Update frozen status
        for section in all_sections:
            if section in sections_with_issues:
                self.frozen_sections[section] = False
                logger.info(f"Not freezing '{section}' section: issues were found.")
            else:
                self.frozen_sections[section] = True
                logger.info(f"Freezing '{section}' section: no issues identified.")
    
        # 4. Assemble high-level feedback string for logging 
        feedback_parts = []
        feedback_parts.append("\n--- Overall Code Review Feedback ---")
        if review_results:
            feedback_parts.append(f"Summary: {review_results.get('summary', 'No summary available')}")
            if review_results.get("critical_issues"):
                feedback_parts.append(f"Critical Issues Count: {len(review_results['critical_issues'])}")
        else:
            feedback_parts.append("Review results unavailable")

        feedback_parts.append("\n--- Overall Validation Feedback ---")
        if execution_results.get("summary"):
            feedback_parts.append(f"Execution/Validation Summary: {execution_results['summary']}")
    
        print("==============================")
        print(f"FROZEN SECTIONS: {self.frozen_sections}")
        return "\n".join(feedback_parts)


    def _evaluate_code(self, attempt: int) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
        """
        Merged step that runs code execution first, then code review with context.
        """
        logger.info("Running merged execution and review evaluation")

        # Run Step 6 (execution + validation)
        success_exec, execution_results = self._step_6_execute_code(attempt)

        # Prepare execution context
        # exec_context = {
        #     "success": execution_results.get("success", False),
        #     "summary": execution_results.get("summary", "No summary."),
        #     "traceback": execution_results.get("stderr", ""),
        #     "validations": execution_results.get("validations", [])
        # }

        exec_context = execution_results

        # Run Step 5 (LLM Review) with execution context
        review_results = self.code_reviewer.review_code(
            code_path=self.generated_code_path,
            requirements_path=self.requirements_path,
            execution_context=exec_context or {}
        )

        # Save results for reference
        self.code_reviewer.save_review_results(review_results, self.review_results_path)
        self.code_execution_agent.save_results(execution_results, self.execution_results_path)



        return success_exec, review_results, execution_results


    def _step_8_output_and_completion(self) -> None:
        """
        Handles the final output and completion of the pipeline.
        """
        logger.info("Starting Step 8: Output and Completion...")
        if self.final_synthetic_data_path and os.path.exists(self.final_synthetic_data_path):
            logger.info(f"Pipeline COMPLETED SUCCESSFULLY!")
            logger.info(f"Final synthetic data file generated at: {os.path.abspath(self.final_synthetic_data_path)}")

            final_destination = os.path.join(self.config["output_dir"], f"{self.config['final_data_filename_prefix']}_final.{self.config['output_format']}")
            try:
                shutil.copy(self.final_synthetic_data_path, final_destination)
                logger.info(f"Final data also copied to: {final_destination}")
            except Exception as e:
                logger.warning(f"Could not copy final data to {final_destination}: {e}")

        else:
            logger.error("Pipeline COMPLETED WITH ERRORS or could not locate the final data file.")
            logger.error(f"Please check logs and intermediate files in {os.path.abspath(self.output_dir)}")

        logger.info("Pipeline run finished.")


    def run_pipeline(self):
        """
        Executes the full data synthesis pipeline from input processing to final output.
        """
        logger.info("========= Starting LLM-Powered Data Synthesizer Pipeline =========")

        start_from_step_4 = self.config.get("start_from_step_4", False)

        if start_from_step_4:
            logger.info("Pipeline configured to start from step 4 (Code Generation)")
            if not self._load_existing_artifacts():
                logger.error("Failed to load required artifacts to start from step 4. Halting pipeline.")
                return
            logger.info("Successfully loaded existing artifacts. Proceeding to step 4.")
            self.current_feedback = None  # Start fresh
        else:
            if not self._step_1_process_inputs():
                logger.error("Halting pipeline due to critical error in Input Processing.")
                return

            if not self._step_2_analyze_metadata():
                logger.error("Halting pipeline due to critical error in Metadata Analysis.")
                return

            if not self._step_3_web_search_if_needed():
                logger.warning("Web search step encountered issues. Proceeding with available information.")

        successful_generation = False

        for attempt in range(1, self.config["max_retries"] + 1):
            logger.info(f"\n--- Iteration {attempt}/{self.config['max_retries']} ---")
            
            success_gen = self._step_4_generate_code(attempt=attempt)

            if not success_gen:
                logger.error("Code generation failed. Retrying with feedback...")
                continue

            success, review_results, execution_results = self._evaluate_code(attempt)

            self.current_feedback = self._gather_feedback(review_results, execution_results)

            if success:
                successful_generation = True
                break
            else:
                logger.warning("Code review or execution failed. Retrying with updated feedback...")

        if successful_generation:
            self._step_8_output_and_completion()
        else:
            logger.error("Pipeline failed after maximum retries.")


if __name__ == "__main__":
    # --- Configuration ---
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not set. Please set it to run the pipeline.")
        google_api_key = "YOUR_API_KEY_HERE" 

    if google_api_key == "YOUR_API_KEY_HERE" and not os.getenv("GOOGLE_API_KEY"):
         print("WARNING: Using a placeholder API key. LLM calls will likely fail.")

    pipeline_config = {
        "google_api_key": google_api_key,
        "google_search_engine_id": os.getenv("GOOGLE_SEARCH_ENGINE_ID"),
        "input_metadata_filepath": "inputs/metadata.json",
        "input_relationships_filepath": "inputs/relationship_out.json",
        "input_samples_filepath": "inputs/fct_retail_casa_sample_data_10000_rows.csv",
        "output_dir": "pipeline_run_outputs",
        "enriched_metadata_filename": "enriched_metadata.json",
        "generated_code_filename": "generated_data_script.py",
        "requirements_doc_filename": "generation_requirements.txt",
        "ordered_columns_filename": "ordered_columns_for_generation.json",
        "search_results_filename": "web_search_results.json",
        "review_results_filename": "code_review.json",
        "execution_results_filename": "code_execution_report.json",
        "final_data_filename_prefix": "synthesized_output_data",
        "original_column_order_filename": "original_column_order.json",
        "output_format": "csv",
        "num_rows": 300,
        "max_retries": 8, 
        "code_execution_timeout": 60, 
        "start_from_step_4": True,
    }

    orchestrator = PipelineOrchestrator(config=pipeline_config)
    orchestrator.run_pipeline()
