## prompt_id: mixed_002 | experiment: decode_ema_score_averaging_ce23db51 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: ema_score_averaging
decode_params: {"ema_beta": 0.5}

### Prompt

I need to analyze a CSV file containing sales data. The file has columns: date, product_id, quantity, price.

Write a Python script that:
1. Loads the data
2. Calculates total revenue per product
3. Finds the top 5 best-selling products
4. Generates a simple monthly sales report

Include comments explaining each section.

### Generated Text

```python
# Import necessary libraries
import pandas as pd

def analyze_sales_data(csv_file_path):
    """
    Analyzes sales data from a CSV file and performs various calculations.
    
    Parameters:
    csv_file_path (str): Path to the CSV file containing sales data.
    """
    
    # 1. Load the data
    # Read the CSV file into a DataFrame
    # The 'parse_dates' parameter ensures the 'date' column is treated as datetime objects
    df = pd.read_csv(csv_file_path, parse_dates=['date'])
    
    # Display first few rows to verify loading
    print("Data loaded successfully. First 5 rows:")
    print(df.head())
    
    # Ensure ensure required columns exist
    required_columns = ['date', 'product_id', 'quantity', 'price']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Expected: {required_columns}")
    
    # Calculate total revenue per product
    # Revenue = quantity * price
    df['revenue'] = df['quantity'] * df['price']
    
    # Group by product_id and sum the revenue
