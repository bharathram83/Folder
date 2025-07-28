import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def process_revenue_data(file_path):
    """
    Ingests the raw revenue data, cleans it, melts it into a long format,
    and identifies acquisition cohorts based on churn logic.
    """
    # Load the data, skipping the first two rows and setting the first column as the index.
    # Specify the encoding to handle potential BOM characters from Excel.
    df = pd.read_csv(file_path, header=1, index_col=0, encoding='utf-8-sig')
    df.index.name = 'Customers'
    df.columns.name = None

    # Filter out any extraneous columns that pandas might read from the CSV
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    
    # Melt the dataframe to transform it from wide to long format
    df_long = df.reset_index().melt(id_vars='Customers', var_name='Month', value_name='Revenue')
    
    # Clean and convert 'Revenue' column to a numeric type
    # This handles '$' signs, commas, and parentheses for negative numbers.
    df_long['Revenue'] = df_long['Revenue'].astype(str).replace({r'\$': '', ',': '', r'\(': '-', r'\)': ''}, regex=True)
    df_long['Revenue'] = pd.to_numeric(df_long['Revenue'], errors='coerce')
    # Address the FutureWarning by assigning the result back instead of using inplace=True
    df_long['Revenue'] = df_long['Revenue'].fillna(0)
    
    # Update the date format to match "Mar-20" style (%b-%y)
    df_long['Month'] = pd.to_datetime(df_long['Month'], format='%b-%y')
    df_long = df_long.sort_values(['Customers', 'Month']).reset_index(drop=True)

    # --- Churn and Re-acquisition Logic ---
    print("Processing churn and re-acquisition logic...")
    processed_data = []
    for customer, group in df_long.groupby('Customers'):
        in_churn_gap = False
        gap_counter = 0
        acquisition_date = None
        
        for _, row in group.iterrows():
            if row['Revenue'] > 0:
                if acquisition_date is None or in_churn_gap:
                    acquisition_date = row['Month']
                    in_churn_gap = False
                
                processed_data.append({
                    'Customer': customer,
                    'Acquisition_Month': acquisition_date,
                    'Month': row['Month'],
                    'Revenue': row['Revenue']
                })
                gap_counter = 0
            else:
                if acquisition_date is not None:
                    gap_counter += 1
                    if gap_counter >= 6:
                        in_churn_gap = True
    
    if not processed_data:
        raise ValueError("No processed data available. Check the input file for revenue entries.")

    return pd.DataFrame(processed_data)

def create_cohort_revenue_table(df):
    """
    Creates a table of total revenue by cohort and cohort age.
    """
    df['Cohort_Age_Months'] = ((df['Month'].dt.year - df['Acquisition_Month'].dt.year) * 12 +
                               (df['Month'].dt.month - df['Acquisition_Month'].dt.month))

    cohort_revenue = df.pivot_table(index='Acquisition_Month',
                                    columns='Cohort_Age_Months',
                                    values='Revenue',
                                    aggfunc='sum')
    
    cohort_revenue.index = cohort_revenue.index.strftime('%Y-%m')
    cohort_revenue.fillna(0, inplace=True)
    return cohort_revenue

def create_nrr_table(cohort_revenue):
    """
    Calculates the Net Revenue Retention (NRR) curves for each cohort.
    """
    if cohort_revenue.shape[1] < 2:
        print("Not enough data to calculate NRR (need at least 2 months of activity).")
        return pd.DataFrame()

    nrr_table = cohort_revenue.iloc[:, 1:].div(cohort_revenue.iloc[:, :-1].values)
    nrr_table.fillna(0, inplace=True)
    nrr_table.replace([np.inf, -np.inf], 0, inplace=True)
    
    return nrr_table

def forecast_revenue(cohort_revenue, nrr_table, new_customer_revenue):
    """
    Forecasts future revenue based on historical NRR and new customer revenue.
    """
    # --- FIX: Calculate average NRR only from active cohorts ---
    # Replace 0s with NaN so they are ignored in the mean() calculation.
    # This prevents churned cohorts from dragging down the average.
    avg_nrr_by_age = nrr_table.replace(0, np.nan).mean()
    
    last_actual_age = cohort_revenue.columns[-1]
    existing_cohorts_forecast = cohort_revenue.copy()
    
    num_forecast_months = len(new_customer_revenue) + 6
    
    for i in range(1, num_forecast_months + 1):
        current_forecast_age = last_actual_age + i
        
        for cohort_date, row in existing_cohorts_forecast.iterrows():
            last_rev_age = row.index[-1]
            last_rev_value = row.iloc[-1]
            
            # If the last revenue was 0, the cohort is churned and will stay 0.
            if last_rev_value == 0:
                forecasted_rev = 0
            else:
                # Get the NRR for the next age. If we don't have historical data for that age,
                # use the last known average NRR as a terminal rate.
                nrr_for_age = avg_nrr_by_age.get(last_rev_age + 1, avg_nrr_by_age.iloc[-1])
                forecasted_rev = last_rev_value * nrr_for_age
            
            existing_cohorts_forecast.loc[cohort_date, current_forecast_age] = forecasted_rev

    last_month_in_data = pd.to_datetime(cohort_revenue.index[-1] + '-01')
    
    new_cohorts_data = []
    for i, new_rev in enumerate(new_customer_revenue):
        cohort_month = last_month_in_data + relativedelta(months=i + 1)
        cohort_data = {'Acquisition_Month': cohort_month.strftime('%Y-%m'), 0: new_rev}
        
        for age in range(1, num_forecast_months - i):
            nrr_for_age = avg_nrr_by_age.get(age, avg_nrr_by_age.iloc[-1])
            cohort_data[age] = cohort_data[age - 1] * nrr_for_age
        new_cohorts_data.append(cohort_data)

    new_cohorts_df = pd.DataFrame(new_cohorts_data).set_index('Acquisition_Month')

    final_forecast, _ = existing_cohorts_forecast.align(new_cohorts_df, join='outer', axis=1, fill_value=0)
    final_forecast.update(new_cohorts_df)
    
    return final_forecast.fillna(0)

def plot_nrr_curves(nrr_df):
    """
    Generates a line chart of the NRR curves for each cohort.
    """
    if nrr_df.empty:
        print("NRR table is empty, skipping NRR curve plot.")
        return

    data_to_plot = nrr_df.T
    
    fig, ax = plt.subplots(figsize=(18, 10))
    data_to_plot.plot(kind='line', ax=ax, alpha=0.8, marker='o', markersize=4, linestyle='--')

    ax.set_title('Net Revenue Retention (NRR) Curves by Cohort', fontsize=18, pad=20)
    ax.set_xlabel('Cohort Age (Months)', fontsize=12)
    ax.set_ylabel('Net Revenue Retention Rate', fontsize=12)
    
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.axhline(y=1.0, color='red', linestyle='-', linewidth=1.5, label='100% NRR (Breakeven)')
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Cohorts', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

def plot_revenue_forecast(forecast_df, actuals_df):
    """
    Generates a stacked area chart of the revenue forecast by cohort.
    """
    data_to_plot = forecast_df.T
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.stackplot(data_to_plot.index, data_to_plot.values.T, labels=data_to_plot.columns, alpha=0.8)
    forecast_start_point = len(actuals_df.columns) - 1
    ax.axvline(x=forecast_start_point, color='red', linestyle='--', linewidth=2, label='Forecast Start')

    ax.set_title('Revenue Forecast by Customer Cohort (Actuals vs. Projections)', fontsize=18, pad=20)
    ax.set_xlabel('Months Since First Cohort', fontsize=12)
    ax.set_ylabel('Total Monthly Revenue', fontsize=12)
    
    def currency_formatter(x, pos):
        return f'${x:,.0f}'
    
    formatter = mticker.FuncFormatter(currency_formatter)
    ax.yaxis.set_major_formatter(formatter)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title='Cohorts', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    file_path = '/content/drive/MyDrive/Colab Notebooks/Untitled spreadsheet - Copy of Paygo + Subscription.csv'
    try:
        # --- FIX: Restructured execution flow to prevent plotting from blocking user input ---

        # === STEP 1: All Data Processing and Forecasting First ===
        print("--- Processing Data and Building Forecast ---")
        processed_df = process_revenue_data(file_path)
        cohort_revenue_table = create_cohort_revenue_table(processed_df)
        nrr_cohort_table = create_nrr_table(cohort_revenue_table)

        # Get user input before any plotting occurs
        print("\n--- Forecasting New Customer Revenue ---")
        new_customer_inputs = []
        for i in range(6):
            while True:
                try:
                    default_val = 20000 + i * 2500
                    val_str = input(f"Enter revenue for future month {i+1} (or press Enter for default of ${default_val:,.0f}): ")
                    new_customer_inputs.append(float(val_str) if val_str else default_val)
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
        
        # Now, create the final forecast dataframe
        final_revenue_forecast = forecast_revenue(cohort_revenue_table, nrr_cohort_table, new_customer_inputs)
        print("\n" + "="*80 + "\n")

        # === STEP 2: Print All Tabular Outputs ===
        print("--- Cohort Revenue Table ---")
        print(cohort_revenue_table.to_string(max_cols=12, formatters={col: '{:,.0f}'.format for col in cohort_revenue_table.columns}))
        print("\n" + "="*80 + "\n")

        print("--- Net Revenue Retention (NRR) Curves ---")
        print(nrr_cohort_table.to_string(max_cols=12, formatters={col: '{:.2%}'.format for col in nrr_cohort_table.columns}))
        print("\n" + "="*80 + "\n")

        print("--- Full Revenue Forecast (Actuals + Projections) ---")
        print(final_revenue_forecast.to_string(max_cols=12, formatters={col: '(${:,.0f})'.format if col not in cohort_revenue_table.columns else '{:,.0f}'.format for col in final_revenue_forecast.columns}))
        print("\n" + "="*80 + "\n")
        
        # === STEP 3: Generate All Visualizations Last ===
        print("--- Generating NRR Curves Visualization ---")
        plot_nrr_curves(nrr_cohort_table)
        
        print("\n--- Generating Final Revenue Forecast Visualization ---")
        plot_revenue_forecast(final_revenue_forecast, cohort_revenue_table)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it is in the correct directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
