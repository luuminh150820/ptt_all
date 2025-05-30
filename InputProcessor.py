import json
import pandas as pd #type: ignore
import os
from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np #type: ignore 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataSynthesizer')

class InputProcessor:
    """
    Component responsible for loading and processing input metadata files
    for the LLM-Powered Data Synthesizer pipeline.
    """
    
    def __init__(self):
        self.metadata = None
        self.column_relationships = None
        self.column_statistics = None
        self.sample_data = None
    
    def load_metadata_file(self, filepath: str) -> Dict:
        """
        Load and parse the main metadata JSON file.
        
        Args:
            filepath: Path to the metadata JSON file
            
        Returns:
            Parsed metadata as a dictionary
        """
        logger.info(f"Loading metadata file from {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"Successfully loaded metadata file")
            self.metadata = metadata
            return metadata
        except Exception as e:
            logger.error(f"Error loading metadata file: {str(e)}")
            raise
    
    def load_column_relationships(self, filepath: str) -> Dict:
        """
        Load and parse the column relationships JSON file.
        
        Args:
            filepath: Path to the column relationships JSON file
            
        Returns:
            Parsed relationships as a dictionary
        """
        logger.info(f"Loading column relationships from {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                relationships = json.load(f)
            logger.info(f"Successfully loaded column relationships")
            self.column_relationships = relationships
            return relationships
        except Exception as e:
            logger.error(f"Error loading column relationships: {str(e)}")
            raise
    
    def load_data_samples(self, filepath: str) -> pd.DataFrame:
        """
        Load data samples file (CSV or XLSX).
        
        Args:
            filepath: Path to the data samples file
            
        Returns:
            DataFrame containing the sample data
        """
        logger.info(f"Loading data samples from {filepath}")
        try:
            # Determine file type by extension
            if filepath.lower().endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath)
            else:
                raise ValueError(f"Unsupported file format for {filepath}. Expected .csv, .xlsx, or .xls")
            
            logger.info(f"Successfully loaded data samples with {len(df)} rows and {len(df.columns)} columns")
            self.sample_data = df
            return df
        except Exception as e:
            logger.error(f"Error loading data samples: {str(e)}")
            raise
    
    def calculate_column_statistics(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Calculate statistics for each column in the data samples using BigQuery DataFrames.
        
        Args:
            df: DataFrame containing the sample data
            
        Returns:
            Dictionary mapping column names to their statistics
        """
        logger.info("Calculating column statistics")
        
        try:
            # For this implementation, we'll use pandas operations
            # In a real implementation, you would use BigQuery DataFrames here
            stats = {}
            
            for column in df.columns:
                column_stats = {
                    'total_rows': int(len(df)),
                    'null_percentage': float((df[column].isna().sum() / len(df)) * 100),
                    'num_unique': int(df[column].nunique())
                }
                
                # Get sample values (up to 20 unique)
                unique_values = df[column].dropna().unique()
                sample_values = [str(v) for v in unique_values[:min(20, len(unique_values))].tolist()]
                column_stats['sample_data'] = sample_values
                
                # Determine data type and type-specific statistics
                if pd.api.types.is_numeric_dtype(df[column]):
                    if all(df[column].dropna().apply(lambda x: float(x).is_integer())):
                        column_stats['data_type'] = 'integer'
                    elif df[column].dropna().apply(lambda x: pd.api.types.is_float(x)).all():
                         column_stats['data_type'] = 'float'
                    else:
                         column_stats['data_type'] = 'numeric' 

                    # Convert numeric stats to standard float/int, handle potential NaNs and empty dataframes
                    column_stats['min_val'] = float(df[column].min()) if len(df) > 0 and pd.notna(df[column].min()) else None
                    column_stats['max_val'] = float(df[column].max()) if len(df) > 0 and pd.notna(df[column].max()) else None
                    column_stats['average'] = float(df[column].mean()) if len(df) > 0 and pd.notna(df[column].mean()) else None
                    column_stats['positive_percentage'] = float((df[column] > 0).mean() * 100) if len(df) > 0 and pd.api.types.is_numeric_dtype(df[column]) else 0.0
                    column_stats['zero_percentage'] = float((df[column] == 0).mean() * 100) if len(df) > 0 and pd.api.types.is_numeric_dtype(df[column]) else 0.0

                elif pd.api.types.is_datetime64_any_dtype(df[column]):
                    column_stats['data_type'] = 'date'
                    column_stats['min_val'] = df[column].min().strftime('%Y-%m-%d') if pd.notna(df[column].min()) else None
                    column_stats['max_val'] = df[column].max().strftime('%Y-%m-%d') if pd.notna(df[column].max()) else None

                elif pd.api.types.is_bool_dtype(df[column]):
                    column_stats['data_type'] = 'boolean'
                    column_stats['true_percentage'] = float(df[column].mean() * 100) if pd.notna(df[column].mean()) else 0.0
                
                else:
                    column_stats['data_type'] = 'string'
                    try:
                        column_stats['avg_length'] = float(df[column].astype(str).str.len().mean()) if len(df) > 0 and not df[column].empty else 0.0
                    except Exception:
                         column_stats['avg_length'] = 0.0
                
                # Check if it's categorical (10 or fewer unique values)
                if column_stats['num_unique'] <= 10 and column_stats['data_type'] != 'date' and column_stats['num_unique'] > 1:
                    column_stats['data_type'] = 'categorical'
                    # Calculate value distribution
                    value_counts = df[column].value_counts(normalize=True)
                    column_stats['value_distribution'] = {
                        str(k): float(v * 100) for k, v in value_counts.items()
                    }
                
                stats[column] = column_stats
            
            logger.info(f"Successfully calculated statistics for {len(stats)} columns")
            self.column_statistics = stats
            return stats
        
        except Exception as e:
            logger.error(f"Error calculating column statistics: {str(e)}")
            raise
    
    def merge_metadata_with_statistics(self) -> Dict:
        """
        Merge the metadata information with calculated statistics, prioritizing columns
        found in the sample data.

        Returns:
            Enriched metadata dictionary
        """
        logger.info("Merging metadata with statistics, prioritizing sample data columns")

        if self.column_statistics is None or self.sample_data is None:
            raise ValueError("Column statistics and sample data must be loaded before merging")

        enriched_metadata: Dict[str, Any] = {}
        original_metadata_all_cols_map_normalized: Dict[str, Dict] = {}
        original_col_to_table_map: Dict[str, str] = {} # Map normalized col name to original table name

        if self.metadata:
            for table_name, table_info in self.metadata.items():
                enriched_metadata[table_name] = table_info.copy()
                enriched_metadata[table_name]['columns'] = [] # Start with an empty list of columns

                if 'columns' in table_info:
                    for col in table_info['columns']:
                        col_name = col.get('Column_name')
                        if col_name:
                            normalized_name = col_name.lower().strip()
                            original_metadata_all_cols_map_normalized[normalized_name] = col
                            original_col_to_table_map[normalized_name] = table_name # Store which table this column came from

        inferred_table_name = "InferredTable"
        if inferred_table_name not in enriched_metadata:
             enriched_metadata[inferred_table_name] = {
                 "description": "Table inferred from sample data columns not found in original metadata",
                 "columns": []
             }

        for sample_col_name in self.sample_data.columns:
            normalized_sample_col_name = sample_col_name.lower().strip()
            stats_data = self.column_statistics.get(sample_col_name)

            if stats_data is None:
                logger.error(f"Statistics not found for sample data column '{sample_col_name}'. Skipping.")
                continue

            original_col_info = original_metadata_all_cols_map_normalized.get(normalized_sample_col_name)

            enriched_column_info: Dict[str, Any] = {}
            target_table_name = inferred_table_name # Default target table is the inferred table

            if original_col_info:
                enriched_column_info = original_col_info.copy()
                target_table_name = original_col_to_table_map.get(normalized_sample_col_name, inferred_table_name)

                if 'Column_name' not in enriched_column_info:
                     enriched_column_info['Column_name'] = sample_col_name # Fallback if metadata entry is malformed

                logger.debug(f"Column '{sample_col_name}' found in original metadata ('{target_table_name}'), merging info.")
            else:
                enriched_column_info['Column_name'] = sample_col_name
                enriched_column_info['Description'] = "Inferred from sample data" # Default description
                enriched_column_info['Key_type'] = "null" # Default key type for inferred columns

                logger.warning(f"Column '{sample_col_name}' not found in original metadata. Adding to '{inferred_table_name}' as inferred.")

            enriched_column_info.update(stats_data)

            if target_table_name not in enriched_metadata:
                 logger.error(f"Internal error: Target table '{target_table_name}' not found in enriched_metadata structure. Adding column '{sample_col_name}' to '{inferred_table_name}'.")
                 target_table_name = inferred_table_name
                 if inferred_table_name not in enriched_metadata: # Double check inferred table exists
                      enriched_metadata[inferred_table_name] = {"description": "Inferred table", "columns": []}

            enriched_metadata[target_table_name]['columns'].append(enriched_column_info)

        tables_to_remove = [
            table_name for table_name, table_info in enriched_metadata.items()
            if table_name != inferred_table_name and (not table_info.get('columns') or len(table_info['columns']) == 0)
        ]
        for table_name in tables_to_remove:
            logger.warning(f"Removing table '{table_name}' from enriched metadata as none of its original columns were found in sample data.")
            del enriched_metadata[table_name]

        logger.info(f"Successfully merged metadata with statistics. Enriched metadata contains {sum(len(t.get('columns',[])) for t in enriched_metadata.values())} columns across {len(enriched_metadata)} tables.")
        return enriched_metadata

    def process_all_inputs(self, 
                          metadata_filepath: str, 
                          relationships_filepath: str, 
                          samples_filepath: str) -> Tuple[Dict, Dict, Dict, List[str]]:
        """
        Process all input files and return the enriched data.
        
        Args:
            metadata_filepath: Path to the metadata JSON file
            relationships_filepath: Path to the column relationships JSON file
            samples_filepath: Path to the data samples file
            
        Returns:
            Tuple containing (enriched_metadata, column_relationships, column_statistics)
        """
        logger.info("Processing all input files")
        
        # Load all input files
        self.load_metadata_file(metadata_filepath)
        self.load_column_relationships(relationships_filepath)
        df = self.load_data_samples(samples_filepath)
        
        original_column_order = df.columns.tolist() # <-- Capture the column order
        logger.info(f"Captured original column order from sample data: {original_column_order}")

        # Calculate statistics
        self.calculate_column_statistics(df)
        
        # Merge metadata with statistics
        enriched_metadata = self.merge_metadata_with_statistics()
        
        logger.info("Successfully processed all input files")
        return enriched_metadata, self.column_relationships, self.column_statistics, original_column_order


# Example usage
if __name__ == "__main__":
    processor = InputProcessor()
    
    # Example file paths - these would be provided as inputs
    metadata_path = "metadata.json"
    relationships_path = "relationships_out.json"
    samples_path = "fct_retail_casa_sample_data_10000_rows.csv"
    
    # Process all inputs
    enriched_metadata, relationships, statistics = processor.process_all_inputs(
        metadata_path, relationships_path, samples_path
    )
    
    # Output the enriched metadata for the next component
    output_path = "enriched_metadata.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_metadata, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Enriched metadata saved to {output_path}")