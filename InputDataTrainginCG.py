import pandas as pd
import numpy as np
import dask.dataframe as dd
import dask.diagnostics
import time
from dask import delayed
from dask.diagnostics import ProgressBar
import logging


def simulate_option_price(row, num_simulations=100000):
    S = row['Spot Price']
    K = row['Strike Price']
    r = row['risk_free_interest']
    sigma = row['Volatility']
    T = row['Maturity']

    z = np.random.standard_normal((num_simulations,))

    # Generate stock prices at the expiration time
    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)

    # Calculate the payoffs of the options at expiration
    payoffs = np.maximum(ST - K, 0)

    # Discount the payoffs back to present value
    option_prices = payoffs * np.exp(-r * T)

    # Estimate the option value as the average of the simulated prices
    option_value = np.mean(option_prices)
    return option_value


def calculate_option_value(df):
    return df.apply(simulate_option_price, axis=1)


if __name__ == '__main__':
    chunksize = 1000

    # Read the input dataset from the CSV file in chunks
    data_chunks = pd.read_csv('InputDataSet_CG.csv', chunksize=chunksize)

    start_time = time.time()

    # Configure logging
    logging.basicConfig(filename='execution.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    try:
        # Calculate the option values for each chunk
        option_values = []
        for i, data_chunk in enumerate(data_chunks, 1):
            # Perform the calculation for each chunk
            start_chunk_time = time.time()
            data_chunk['Call_Premium'] = calculate_option_value(data_chunk)
            data_chunk['Call_Premium'] = data_chunk['Call_Premium'].round(4)  # Round off to 4 decimal places
            end_chunk_time = time.time()

            # Append the chunk to the list of option values
            option_values.append(data_chunk)

            # Update progress and time
            elapsed_time = end_chunk_time - start_chunk_time
            logging.info(f"Chunk {i} processed. Time: {elapsed_time:.2f} seconds")

    except Exception as e:
        # Log the exception to the error log file
        logging.error(f"An exception occurred: {str(e)}", exc_info=True)

    else:
        # Concatenate the chunks into a single DataFrame
        result = pd.concat(option_values)

        # Save the updated DataFrame to a new CSV file
        result.to_csv('InputDataSet_updated.csv', index=False)

    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Total Execution time: {execution_time} seconds")
