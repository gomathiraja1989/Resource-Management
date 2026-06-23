import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import chardet
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_dir: str = "DataSets"):
        """Initialize the DataLoader with the directory containing CSV files.
        
        Args:
            data_dir (str): Path to the directory containing the CSV files.
        """
        self.data_dir = Path(data_dir)
        self.data: Dict[str, pd.DataFrame] = {}
        self.merged_data: Optional[pd.DataFrame] = None
        
    def detect_encoding(self, file_path: Path) -> Tuple[str, float]:
        """Detect the encoding of a file and return the encoding with confidence."""
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10k bytes to guess encoding
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8', result.get('confidence', 0.0)
        
    def try_read_csv(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Try reading a CSV file with different encodings and parameters."""
        # List of encodings to try (in order of preference)
        encodings = [
            'utf-8', 'latin-1', 'iso-8859-1', 'windows-1252',
            'cp1252', 'utf-16', 'utf-16le', 'utf-16be'
        ]
        
        # First try to detect encoding
        try:
            detected_encoding, confidence = self.detect_encoding(file_path)
            if detected_encoding and confidence > 0.7:  # Only use if confident
                if detected_encoding not in encodings:
                    encodings.insert(0, detected_encoding)
        except Exception as e:
            logger.warning(f"Error detecting encoding for {file_path.name}: {str(e)}")
        
        # Try different reading strategies
        strategies = [
            # Try with python engine first (more lenient)
            {'engine': 'python', 'on_bad_lines': 'warn', 'dtype': str, 'low_memory': False},
            # Try with c engine (faster but less forgiving)
            {'engine': 'c', 'error_bad_lines': False, 'dtype': str, 'low_memory': False},
            # Try with no type inference
            {'engine': 'python', 'dtype': str, 'on_bad_lines': 'skip'}
        ]
        
        for encoding in encodings:
            for strategy in strategies:
                try:
                    logger.info(f"Trying to read {file_path.name} with {encoding} encoding and {strategy}")
                    
                    # Special handling for utf-16 files
                    if 'utf-16' in encoding:
                        # For utf-16, we need to read the file in binary mode first
                        with open(file_path, 'rb') as f:
                            # Skip BOM if present
                            if f.read(2) != b'\xff\xfe':
                                f.seek(0)
                            # Read the rest of the file
                            content = f.read()
                            
                        # Decode with the specified encoding
                        text = content.decode(encoding)
                        
                        # Read the CSV from the decoded text
                        from io import StringIO
                        df = pd.read_csv(
                            StringIO(text),
                            **{k: v for k, v in strategy.items() if k != 'low_memory'}
                        )
                    else:
                        # For other encodings, read directly
                        df = pd.read_csv(
                            file_path,
                            encoding=encoding,
                            **{k: v for k, v in strategy.items() if k != 'low_memory' or k in ('dtype', 'on_bad_lines')}
                        )
                    
                    # If we got here, the read was successful
                    if not df.empty:
                        logger.info(f"Successfully read {file_path.name} with {encoding} encoding")
                        return df
                        
                except Exception as e:
                    logger.warning(f"Failed with {encoding} and {strategy}: {str(e)}")
                    continue
                
        return None
    
    def load_data(self) -> None:
        """Load all CSV files from the data directory."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        total_files = len(csv_files)
        success_count = 0
        
        for idx, file_path in enumerate(csv_files, 1):
            state_name = self._extract_state_name(file_path.stem)
            logger.info(f"Processing file {idx}/{total_files}: {file_path.name}")
            
            try:
                # Try to read the CSV file
                df = self.try_read_csv(file_path)
                
                if df is None or df.empty:
                    logger.error(f"Failed to read {file_path.name}: Could not decode with any encoding")
                    continue
                
                # Clean and standardize the data
                df = clean_column_names(df)
                
                # Rename columns to standard names
                column_mapping = {
                    'main_workers_total_persons': 'main_workers_total',
                    'main_workers_total_males': 'main_workers_male',
                    'main_workers_total_females': 'main_workers_female',
                    'marginal_workers_total_persons': 'marginal_workers_total',
                    'marginal_workers_total_males': 'marginal_workers_male',
                    'marginal_workers_total_females': 'marginal_workers_female',
                    'nic_name': 'nic_name',
                    'group': 'group',
                    'class': 'class',
                    'division': 'division',
                    'district_code': 'district_code',
                }
                
                # Rural/Urban column mappings
                rural_urban_mapping = {
                    'main_workers_rural_persons': 'main_workers_rural_persons',
                    'main_workers_rural_males': 'main_workers_rural_male',
                    'main_workers_rural_females': 'main_workers_rural_female',
                    'main_workers_urban_persons': 'main_workers_urban_persons',
                    'main_workers_urban_males': 'main_workers_urban_male',
                    'main_workers_urban_females': 'main_workers_urban_female',
                    'marginal_workers_rural_persons': 'marginal_workers_rural_persons',
                    'marginal_workers_rural_males': 'marginal_workers_rural_male',
                    'marginal_workers_rural_females': 'marginal_workers_rural_female',
                    'marginal_workers_urban_persons': 'marginal_workers_urban_persons',
                    'marginal_workers_urban_males': 'marginal_workers_urban_male',
                    'marginal_workers_urban_females': 'marginal_workers_urban_female',
                }
                column_mapping.update(rural_urban_mapping)
                
                df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
                
                # Extract state from indiastates column or use state_name from filename
                if 'indiastates' in df.columns:
                    # Extract just the state name from values like "STATE - RAJASTHAN" or "Union Territory - ..."
                    df['state'] = df['indiastates'].astype(str).str.extract(r'(?:STATE|Union Territory)\s*-\s*(.+)', expand=False)
                    # Forward fill and backward fill to propagate state name to all rows
                    df['state'] = df['state'].ffill().bfill()
                    # Fallback: use the state_name derived from filename
                    if df['state'].isna().iloc[0] if len(df) > 0 else True:
                        df['state'] = state_name
                else:
                    df['state'] = state_name
                
                df['state'] = df['state'].astype(str).str.strip().str.upper()
                df['state'] = df['state'].str.replace(r'\(.*\)', '', regex=True).str.strip()
                
                # Convert numeric columns to appropriate types
                numeric_cols = [
                    'main_workers_total', 'main_workers_male', 'main_workers_female',
                    'marginal_workers_total', 'marginal_workers_male', 'marginal_workers_female',
                    'main_workers_rural_persons', 'main_workers_rural_male', 'main_workers_rural_female',
                    'main_workers_urban_persons', 'main_workers_urban_male', 'main_workers_urban_female',
                    'marginal_workers_rural_persons', 'marginal_workers_rural_male', 'marginal_workers_rural_female',
                    'marginal_workers_urban_persons', 'marginal_workers_urban_male', 'marginal_workers_urban_female'
                ]
                
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col].astype(str).str.replace('[^\d.]', '', regex=True), errors='coerce')
                        df[col] = df[col].fillna(0).astype(int)
                
                # Calculate derived columns
                if 'main_workers_total' in df.columns and 'marginal_workers_total' in df.columns:
                    df['total_workers'] = df['main_workers_total'] + df['marginal_workers_total']
                if 'main_workers_male' in df.columns and 'marginal_workers_male' in df.columns:
                    df['total_workers_male'] = df['main_workers_male'] + df['marginal_workers_male']
                if 'main_workers_female' in df.columns and 'marginal_workers_female' in df.columns:
                    df['total_workers_female'] = df['main_workers_female'] + df['marginal_workers_female']
                
                # Store the data
                self.data[state_name] = df
                success_count += 1
                logger.info(f"Successfully loaded {file_path.name} as {state_name}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {str(e)}", exc_info=True)
        
        logger.info(f"Data loading complete. Successfully loaded {success_count} out of {total_files} files.")
        
        if not self.data:
            raise ValueError("Failed to load any data files. Please check the logs for errors.")
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to a common format."""
        if df is None or df.empty:
            return df
            
        # Clean and standardize column names (same as clean_column_names but inline)
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r'[^\w\s]', '', regex=True)
            .str.replace(r'\s+', '_', regex=True)
        )
        
        # Create a mapping of actual column names to standard names
        column_mapping = {
            'main_workers_total_persons': 'main_workers_total',
            'main_workers_total_males': 'main_workers_male',
            'main_workers_total_females': 'main_workers_female',
            'marginal_workers_total_persons': 'marginal_workers_total',
            'marginal_workers_total_males': 'marginal_workers_male',
            'marginal_workers_total_females': 'marginal_workers_female',
            'state': 'state',
            'indiastates': 'state',
            'nic_name_industry_sector_industry_name_sector_name_nic_name': 'nic_name',
            'group': 'group',
            'class': 'class',
            'division': 'division',
            'district_code': 'district_code',
        }
        
        # Rural/Urban column mappings
        rural_urban_mapping = {
            'main_workers_rural_persons': 'main_workers_rural_persons',
            'main_workers_rural_males': 'main_workers_rural_male',
            'main_workers_rural_females': 'main_workers_rural_female',
            'main_workers_urban_persons': 'main_workers_urban_persons',
            'main_workers_urban_males': 'main_workers_urban_male',
            'main_workers_urban_females': 'main_workers_urban_female',
            'marginal_workers_rural_persons': 'marginal_workers_rural_persons',
            'marginal_workers_rural_males': 'marginal_workers_rural_male',
            'marginal_workers_rural_females': 'marginal_workers_rural_female',
            'marginal_workers_urban_persons': 'marginal_workers_urban_persons',
            'marginal_workers_urban_males': 'marginal_workers_urban_male',
            'marginal_workers_urban_females': 'marginal_workers_urban_female',
        }
        
        column_mapping.update(rural_urban_mapping)
        
        # Apply the renaming
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Calculate derived columns
        if 'main_workers_total' in df.columns and 'marginal_workers_total' in df.columns:
            df['total_workers'] = df['main_workers_total'] + df['marginal_workers_total']
            
        if 'main_workers_male' in df.columns and 'marginal_workers_male' in df.columns:
            df['total_workers_male'] = df['main_workers_male'] + df['marginal_workers_male']
            
        if 'main_workers_female' in df.columns and 'marginal_workers_female' in df.columns:
            df['total_workers_female'] = df['main_workers_female'] + df['marginal_workers_female']
            
        # Ensure state is a string and clean it up
        if 'state' in df.columns:
            df['state'] = df['state'].astype(str).str.strip().str.upper()
            # Remove any state codes or extra information in parentheses
            df['state'] = df['state'].str.replace(r'\(.*\)', '', regex=True).str.strip()
        
        # Convert numeric columns to appropriate types
        numeric_cols = [
            'main_workers_total', 'main_workers_male', 'main_workers_female',
            'marginal_workers_total', 'marginal_workers_male', 'marginal_workers_female',
            'total_workers', 'total_workers_male', 'total_workers_female',
            'main_workers_rural_persons', 'main_workers_rural_male', 'main_workers_rural_female',
            'main_workers_urban_persons', 'main_workers_urban_male', 'main_workers_urban_female',
            'marginal_workers_rural_persons', 'marginal_workers_rural_male', 'marginal_workers_rural_female',
            'marginal_workers_urban_persons', 'marginal_workers_urban_male', 'marginal_workers_urban_female'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                # Convert to string first to handle any non-numeric values
                df[col] = pd.to_numeric(df[col].astype(str).str.replace('[^\d.]', '', regex=True), errors='coerce')
                # Fill NaN with 0 for numeric columns
                df[col] = df[col].fillna(0).astype(int)
        
        return df
    
    def _extract_state_name(self, filename: str) -> str:
        """Extract state name from filename."""
        # Remove common prefixes and suffixes
        name = re.sub(r'^DDW_B18(?:st|sc)?_\\d{4}_NIC_FINAL_STATE_', '', filename)
        name = re.sub(r'-2011$', '', name)
        # Convert to title case and replace underscores with spaces
        name = name.replace('_', ' ').title()
        return name
    
    def merge_data(self) -> None:
        """Merge all loaded data into a single DataFrame."""
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")
            
        self.merged_data = pd.concat(self.data.values(), ignore_index=True)
        print(f"Merged data shape: {self.merged_data.shape}")
        print("Columns in merged data:", self.merged_data.columns.tolist())
    
    def get_merged_data(self) -> pd.DataFrame:
        """Get the merged DataFrame."""
        if self.merged_data is None:
            self.merge_data()
        return self.merged_data
    
    def get_state_data(self, state_name: str) -> pd.DataFrame:
        """Get data for a specific state."""
        return self.data.get(state_name)
    
    def get_available_states(self) -> List[str]:
        """Get list of available states in the loaded data."""
        return list(self.data.keys())

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names by converting to lowercase and replacing spaces with underscores."""
    # Ensure all column names are strings
    df.columns = df.columns.astype(str)
    
    # Clean column names
    df.columns = (
        df.columns
        .str.strip()  # Remove leading/trailing whitespace
        .str.lower()  # Convert to lowercase
        .str.replace(r'[^\w\s]', '', regex=True)  # Remove special characters
        .str.replace(r'\s+', '_', regex=True)  # Replace spaces with underscores
    )
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the merged DataFrame."""
    if df is None or df.empty:
        return pd.DataFrame()
        
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Clean column names
    df = clean_column_names(df)
    
    # Handle missing values
    df = df.dropna(how='all')
    
    if df.empty:
        return df
    
    # Define potential numeric columns
    numeric_cols = [
        'main_workers_total', 'main_workers_male', 'main_workers_female',
        'marginal_workers_total', 'marginal_workers_male', 'marginal_workers_female',
        'total_workers', 'total_workers_male', 'total_workers_female',
        'main_workers_rural_persons', 'main_workers_rural_male', 'main_workers_rural_female',
        'main_workers_urban_persons', 'main_workers_urban_male', 'main_workers_urban_female',
        'marginal_workers_rural_persons', 'marginal_workers_rural_male', 'marginal_workers_rural_female',
        'marginal_workers_urban_persons', 'marginal_workers_urban_male', 'marginal_workers_urban_female',
        'persons', 'males', 'females', 'workers', 'non_workers'
    ]
    
    # Find which numeric columns actually exist in the dataframe
    existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    # Convert numeric columns
    for col in existing_numeric_cols:
        try:
            # First convert to string, then clean and convert to numeric
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r'[^\d.]', '', regex=True)  # Remove non-numeric characters except decimal point
                .replace('', '0')  # Replace empty strings with '0'
                .astype(float)  # Convert to float
                .fillna(0)  # Fill any remaining NaNs with 0
                .astype(int)  # Convert to integer
            )
        except Exception as e:
            logger.warning(f"Error converting column '{col}' to numeric: {str(e)}")
    
    # Ensure state column exists and is string type
    if 'state' in df.columns:
        df['state'] = df['state'].astype(str).str.strip()
    
    return df