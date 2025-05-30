import json
import os
import logging
import shutil
from typing import Dict, Any, Optional, Tuple, List

# Assuming other agent classes are in their respective .py files
# and are importable (e.g., they are in the same directory or PYTHONPATH)
from InputProcessor import InputProcessor           #type: ignore
from LLMAnalyzer import LLMAnalyzer                 #type: ignore
from WebSearchAgent import WebSearchAgent           #type: ignore
from CodeGenerator import CodeGenerator             #type: ignore
from CodeReviewer import CodeReviewer               #type: ignore
from CodeExecutionAgent import CodeExecutionAgent   #type: ignore

# Configure logging for the orchestrator
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Outputs to console
        # You can add logging.FileHandler("orchestrator.log") here if needed
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

        # Create output directory if it doesn't exist
        self.output_dir = config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory set to: {os.path.abspath(self.output_dir)}")

        # Initialize agents
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
        self.last_review_results: Optional[Dict] = None # Added for feedback loop
        self.last_execution_results: Optional[Dict] = None # Added for feedback loop
        self.final_synthetic_data_path: Optional[str] = None

    def _load_existing_artifacts(self):
        """
        Load existing artifacts needed for step 4 onwards when starting from a later step.
        Returns True if successful, False otherwise.
        """
        logger.info("Loading existing artifacts for starting from step 4...")
        try:
            # Check if required files exist
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
                # Load original column order
                with open(self.original_column_order_path, 'r', encoding='utf-8') as f:
                    self.original_column_order = json.load(f)
                    logger.info("Original column order loaded successfully.")
        
            # Load ordered columns
            with open(self.ordered_columns_path, 'r', encoding='utf-8') as f:
                self.ordered_columns = json.load(f)
                logger.info("Ordered columns loaded successfully.")
            
            # Load enriched metadata (might be needed by some steps)
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
            
            return True
        except Exception as e:
            logger.error(f"Failed to load existing artifacts: {e}", exc_info=True)
            return False

    def _step_1_process_inputs(self) -> bool:
        """
        Loads and processes initial metadata, relationships, and sample data.
        Saves enriched metadata.
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
        Saves requirements document and ordered columns list.
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
        Saves search results.
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
            # Create an empty search_results.json
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
                # Create empty if not found, so CodeGenerator doesn't fail on missing file
                with open(self.search_results_path, 'w', encoding='utf-8') as f:
                    json.dump({}, f)

            logger.info("Step 3: Web Search COMPLETED.")
            return True
        except Exception as e:
            logger.error(f"Step 3: Web Search FAILED. Error: {e}", exc_info=True)
            # Create empty if error, so CodeGenerator doesn't fail on missing file
            with open(self.search_results_path, 'w', encoding='utf-8') as f:
                json.dump({}, f)
            return False 

    def _step_4_generate_code(self, attempt: int, previous_code: Optional[str] = None, original_column_order: Optional[List[str]] = None) -> bool: 
        """
        Generates Python code for data synthesis using LLM.
        Incorporates feedback from previous attempts if any.
        Saves the generated code.
        """
        logger.info(f"Starting Step 4: Code Generation (Attempt {attempt})...")
        if not self.requirements_path or not self.ordered_columns_path:
            logger.error("Step 4: Missing requirements or ordered columns path.")
            return False

        try:
            # Ensure search_results_path exists
            if not os.path.exists(self.search_results_path):
                logger.warning(f"Search results file not found at {self.search_results_path}. Creating an empty one.")
                with open(self.search_results_path, 'w', encoding='utf-8') as f:
                    json.dump({}, f)

            generated_code_str = self.code_generator.generate_code(
                requirements_path=self.requirements_path,
                ordered_columns_path=self.ordered_columns_path,
                search_results_path=self.search_results_path,
                output_format=self.config["output_format"],
                num_rows=self.config["num_rows"],
                feedback=self.current_feedback, # Pass current feedback
                previous_code=previous_code, # Pass the previous code
                original_column_order=original_column_order,
                frozen_columns=self.last_review_results.get("passed_columns", []) if self.last_review_results else []
            )
            self.code_generator.save_generated_code(generated_code_str, self.generated_code_path)
            logger.info(f"Generated code saved to {self.generated_code_path}")
            logger.info(f"Step 4: Code Generation (Attempt {attempt}) COMPLETED.")
            return True
        except Exception as e:
            logger.error(f"Step 4: Code Generation (Attempt {attempt}) FAILED. Error: {e}", exc_info=True)
            return False

    def _step_5_review_code(self, attempt: int) -> Tuple[bool, Optional[Dict]]:
        """
        Performs static analysis/review of the generated code.
        Saves review results.
        Returns (is_acceptable, review_results_dict)
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
        Saves execution and validation results.
        Returns (is_successful, execution_results_dict)
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
            self.code_execution_agent.save_results(execution_results, self.execution_results_path)
            logger.info(f"Execution and validation results saved to {self.execution_results_path}")

            is_successful = execution_results.get("overall_success", False)

            if is_successful:
                logger.info(f"Step 6: Code Execution & Validation (Attempt {attempt}) SUCCEEDED.")
                if execution_results.get("validation") and execution_results["validation"].get("data_path_validated"):
                    self.final_synthetic_data_path = execution_results["validation"]["data_path_validated"]
                else: 
                    potential_path = os.path.join(self.output_dir, f"{self.config['final_data_filename_prefix']}.{self.config['output_format']}")
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
        feedback_parts = []

        # --- Frozen Column Tracker from Review ---
        frozen_columns = set(review_results.get("passed_columns", []))
        if frozen_columns:
            sorted_cols = sorted(frozen_columns)
            feedback_parts.append(
                f"The following columns passed validation and should not be modified: {', '.join(sorted_cols)}."
            )

        # --- Code Review Section --- 
        feedback_parts.append("\n--- Code Review Feedback ---")
        summary = review_results.get("summary", "")
        if summary:
            feedback_parts.append(f"Summary: {summary}")

        crit_issues = review_results.get("critical_issues", [])
        non_crit_issues = review_results.get("non_critical_issues", [])

        if crit_issues:
            feedback_parts.append("\nCritical Issues:")
            for issue in crit_issues:
                feedback_parts.append(f"- {issue.get('description', '')} → {issue.get('recommendation', '')}")
                issue_traceback = issue.get('traceback')
                if issue_traceback:
                    feedback_parts.append(f"  Traceback: {issue_traceback}")

        if non_crit_issues:
            feedback_parts.append("\nNon-Critical Issues:")
            for issue in non_crit_issues:
                feedback_parts.append(f"- {issue.get('description', '')} → {issue.get('recommendation', '')}")

        # --- Validation Feedback Section ---
        feedback_parts.append("\n--- Validation Feedback ---")
        val_summary = execution_results.get("summary", "No validation summary available.")
        feedback_parts.append(f"Validation Summary: {val_summary}")

        validations = execution_results.get("validations", [])
        for val in validations:
            if not val.get("passed", True):
                feedback_parts.append(f"- {val.get('name')} on {val.get('column')}: {val.get('details')}")

        return "\n".join(feedback_parts)


    def _evaluate_code(self, attempt: int) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
        """
        Merged step that runs code execution first, then code review with context.
        Returns:
        success (bool): True if execution passed
        review_results: structured LLM review output
        execution_results: structured execution/validation output
        """
        logger.info("Running merged execution and review evaluation")

        # Run Step 6 (execution + validation)
        success_exec, execution_results = self._step_6_execute_code(attempt)

        # Prepare execution context
        exec_context = {
            "success": execution_results.get("success", False),
            "summary": execution_results.get("summary", "No summary."),
            "traceback": execution_results.get("stderr", ""),
            "validations": execution_results.get("validations", [])
        }

        # Run Step 5 (LLM Review) with execution context
        review_results = self.code_reviewer.review_code(
            code_path=self.generated_code_path,
            requirements_path=self.requirements_path,
            execution_context=exec_context
        )

        # Save results for reference
        self.code_reviewer.save_review_results(review_results)
        self.code_execution_agent.save_results(execution_results)

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

            previous_code_content = None
            if attempt > 1 and os.path.exists(self.generated_code_path):
                with open(self.generated_code_path, 'r', encoding='utf-8') as f:
                    previous_code_content = f.read()

            success_gen = self._step_4_generate_code(
                attempt=attempt,
                previous_code=previous_code_content,
                original_column_order=self.original_column_order
            )

            if not success_gen:
                logger.error("Code generation failed. Retrying with feedback...")
                continue

            success, review_results, execution_results = self._evaluate_code(attempt)

            self.last_review_results = review_results
            self.last_execution_results = execution_results

            self.current_feedback = self._gather_feedback(review_results, execution_results)

            if success:
                successful_generation = True
                break
            else:
                logger.warning("Retrying with updated feedback...")

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
