#%%
import math
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import time
import psutil
import os
case_name='potatoes production in Honduras'
class Errors:
    def __init__(self):
        self.E = 0.0
        self.D = 0.0
        self.minFH = 0  # reasonable forecasting horizon

class Solution:
    def __init__(self):
        self.a = None
        self.z = None
        self.Z = 0.0
        self.py = None

# Constants
LARGE = 0x10000

# Function pointers are represented as elements in a list
G = [None, None]  # Array for function pointers, now only for G1 and G2



def G1(x):
    return x

def G2(x):
    return x * x

# Update GForming to include only G1 and G2
def GForming():
    G[0] = G1
    G[1] = G2

def SSTForming(_Y):
    _SST = [[0.0 for _ in range(summs_count * 2 + 2)] for _ in range(summs_count + 1)]

    for i in range(1, summs_count + 1):
        for j in range(1, summs_count + 1):
            for k in range(2, impl_len + 1):
                # Adjust the indices to be 0-based for accessing G
                _SST[i][j] += G[i-1](_Y[k - 1]) * G[j-1](_Y[k - 1])

        for j in range(1, summs_count + 1):
            _SST[i][summs_count + j] = 0.0
        _SST[i][summs_count + i] = 1.0

    return _SST



def JGTransforming(nn, _SST):
    for iter_first in range(1, nn + 1):
        # Find Lead Row
        mm = iter_first
        M = abs(_SST[iter_first][iter_first])

        for iter_second in range(iter_first + 1, nn + 1):
            Mi = abs(_SST[iter_second][iter_first])
            if Mi > M:
                mm = iter_second
                M = Mi

        # Swapping of current N-th and lead mm-th rows
        _SST[iter_first], _SST[mm] = _SST[mm], _SST[iter_first]

        # Normalization of the current row
        Temp = _SST[iter_first][iter_first]
        for iter_second in range(iter_first, 2 * nn + 1):
            _SST[iter_first][iter_second] /= Temp

        # Orthogonalize the Current Column
        for iter_second in range(1, iter_first):
            Temp = _SST[iter_second][iter_first]
            for iter_third in range(iter_first, 2 * nn + 1):
                _SST[iter_second][iter_third] -= _SST[iter_first][iter_third] * Temp

        for iter_second in range(iter_first + 1, nn + 1):
            Temp = _SST[iter_second][iter_first]
            for iter_third in range(iter_first, 2 * nn + 1):
                _SST[iter_second][iter_third] -= _SST[iter_first][iter_third] * Temp

        # Printing the matrix
        print('\nMatrix SST^-1\n')
        for iter_first in range(1, nn + 1):
            print('\n', iter_first, '\t', end='')
            for iter_third in range(1, 2 * nn + 1):
                print(_SST[iter_first][iter_third], '\t', end='')


def P1Forming(_Y, _SST):
    _P1 = [[0.0 for _ in range(summs_count + 1)] for _ in range(impl_len + 2)]

    for t in range(2, impl_len + 1):
        for j in range(1, summs_count + 1):
            _P1[t][j] = 0.0
            for k in range(1, summs_count + 1):
                # Adjust the index to be 0-based for accessing G
                _P1[t][j] += G[k-1](_Y[t - 1]) * _SST[k][summs_count + j]

    return _P1



def PForming(_Y, _P1):
    _P = [[0.0 for _ in range(impl_len + 2)] for _ in range(impl_len + 2)]

    for iter_first in range(2, impl_len + 1):  # Adjust to start from 2
        for iter_second in range(2, impl_len + 1):  # Adjust to start from 2
            _P[iter_first][iter_second] = 0.0

            for iter_third in range(1, summs_count + 1):
                # Adjust the index to be 0-based for accessing G
                _P[iter_first][iter_second] -= G[iter_third-1](_Y[iter_second - 1]) * _P1[iter_first][iter_third]

            if iter_first == iter_second:
                _P[iter_first][iter_first] += 1.0

    return _P



def PrGradForming(_Y, _P):
    # Initialize the _Prgrad array
    _Prgrad = [0.0 for _ in range(impl_len + 2)]
    _grad = [0.0 for _ in range(impl_len + 2)]

    # Copying _Y values to _grad
    for i in range(1, impl_len + 2):
        _grad[i] = _Y[i]

    for iter_first in range(2, impl_len + 1):  # Adjusted to start from 2
        _Prgrad[iter_first] = 0.0
        for iter_second in range(2, impl_len + 1):  # Adjusted to start from 2
            _Prgrad[iter_first] += _P[iter_first][iter_second] * _grad[iter_second]

    # Printing the results
    print('\ni   grad[i]   Prgrad[i]    p[i]  \n', end='')
    for iter_first in range(2, impl_len + 1):  # Adjusted range
        print(f'\n{iter_first}\t{_grad[iter_first]}\t{_Prgrad[iter_first]}\t', end='')

    return _Prgrad


def DualWLDMSolution(_w, _p, _Prgrad):
    Al = LARGE
    Alc = 0

    for iter_first in range(2, impl_len + 1):  # Adjusted to start from 2
        _w[iter_first] = 0

    iter_first = 0
    while iter_first < impl_len - summs_count - 1:  # Adjusted the condition
        Al = LARGE
        for iter_second in range(2, impl_len + 1):  # Adjusted to start from 2
            if abs(_w[iter_second]) == _p[iter_second]:
                continue
            else:
                if _Prgrad[iter_second] > 0:
                    Alc = (_p[iter_second] - _w[iter_second]) / _Prgrad[iter_second]
                elif _Prgrad[iter_second] < 0:
                    Alc = (-_p[iter_second] - _w[iter_second]) / _Prgrad[iter_second]

                if Alc < Al:
                    Al = Alc

        for iter_second in range(2, impl_len + 1):  # Adjusted to start from 2
            if abs(_w[iter_second]) != _p[iter_second]:
                _w[iter_second] += Al * _Prgrad[iter_second]
                if abs(_w[iter_second]) == _p[iter_second]:
                    iter_first += 1


def PrimalWLDMSolution(_Y, _SST, _w, _p, _a, _z):
    lc_r = [0 for _ in range(summs_count + 1)]  # Ordinal numbers of the basic equations
    lc_ri = 0  # The amount of basic equations of the primal problem

    for iter_first in range(2, impl_len + 1):  # Adjusted to start from 2
        if abs(_w[iter_first]) != _p[iter_first]:
            lc_ri += 1
            lc_r[lc_ri] = iter_first

    for iter_first in range(1, lc_ri + 1):
        for iter_second in range(1, lc_ri + 1):
            # Adjust the index to be 0-based for accessing G
            _SST[iter_first][iter_second] = G[iter_second - 1](_Y[lc_r[iter_first] - 1])

        _SST[iter_first][lc_ri + 1] = _Y[lc_r[iter_first]]

    JGTransforming(lc_ri, _SST)

    for iter_first in range(1, lc_ri + 1):
        _a[iter_first] = _SST[iter_first][lc_ri + 1]
        _z[lc_r[iter_first]] = 0



def GLDMEstimator(_Y):
    lc_w = [0.0 for _ in range(impl_len + 2)]  # WLDM weights
    lc_p = [1.0 for _ in range(impl_len + 2)]  # GLDM weights

    lc_a1 = [0.0 for _ in range(summs_count + 1)]
    lc_a = [0.0 for _ in range(summs_count + 1)]  # Identified parameters
    lc_z = [0.0 for _ in range(impl_len + 2)]  # WLDM approximation errors

    lc_SST = SSTForming(_Y)  # Matrix for J-G transforming
    JGTransforming(summs_count, lc_SST)
    lc_P1 = P1Forming(_Y, lc_SST)  # It is used for P calculation
    lc_P = PForming(_Y, lc_P1)  # Projection matrix
    lc_Prgrad = PrGradForming(_Y, lc_P)  # Projection of the gradient

    Z = d = 0.0
    while True:
        for i in range(1, summs_count + 1):
            lc_a1[i] = lc_a[i]

        for i in range(1, impl_len + 1):
            lc_p[i] = 1.0 / (1.0 + lc_z[i] * lc_z[i])

        for i in range(1, impl_len + 1):
            lc_w[i] = 0.0

        DualWLDMSolution(lc_w, lc_p, lc_Prgrad)
        PrimalWLDMSolution(_Y, lc_SST, lc_w, lc_p, lc_a, lc_z)

        Z = lc_z[1] = 0.0
        for i in range(2, impl_len + 1):
            lc_z[i] = _Y[i]
            for j in range(1, summs_count + 1):
                lc_z[i] -= lc_a[j] * G[j-1](_Y[i - 1])  # Adjusted for first-order model
            Z += abs(lc_z[i])

        d = max([abs(lc_a[i] - lc_a1[i]) for i in range(1, summs_count + 1)])
        if d < 0.5:  # some_tolerance_value:  # Define some_tolerance_value as per your requirements
            break

    Sol = Solution()
    Sol.a = lc_a
    Sol.z = lc_z
    Sol.Z = Z

    return Sol


def ForecastingEst(Y, Sol):
    PY = [[] for _ in range(len(Y) + 2)]
    FH = [0] * (len(Y) + 2)
    e = Errors()
    E = [0] * (len(Y) + 2)
    D = [0] * (len(Y) + 2)
    n = len(Sol.a)
    m = len(Y) - 1

    for i in range(len(Y) + 2):
        PY[i] = [0] * (len(Y) + 2)

    T, t = 0, 0
    St = 0
    Et = 0

    while St < m:
        St += 1
        PY[St][0] = Y[St]
        t = 0

        while True:
            t += 1

            # Check if 't' is within bounds for PY[St]
            if t >= len(PY[St]):
                break

            py = 0
            for j in range(1, n):
                # Adjust the index to be 0-based for accessing G
                A1 = G[j-1](PY[St][t - 1])  # Adjusted for first-order
                py += Sol.a[j] * A1

            # Now safe to assign since we've checked the bounds
            PY[St][t] = py

            if St + t < len(Y) and t < len(PY[St]):
                if abs(PY[St][t] - Y[St + t]) > Sol.Z:
                    break

        FH[St] = t
        if St + FH[St] >= m:
            break

    e.minFH = FH[St]
    for t in range(2, St):
        if FH[t] < e.minFH:
            e.minFH = FH[t]

    e.E, e.D = 0, 0
    for t in range(2, e.minFH + 1):
        # Check if indices are within the bounds
        if t + St < len(Y) and t < len(PY[St]):
            e.D += abs(Y[t + St] - PY[St][t])
            e.E += (Y[t + St] - PY[St][t])
        else:
            break  # Break the loop if index out of bounds

    if e.minFH > 0:  # Avoid division by zero
        e.E /= e.minFH
        e.D /= e.minFH
    return e


def calculate_rmse(actual, predicted):
    return np.sqrt(((np.array(actual) - np.array(predicted)) ** 2).mean())

def calculate_r_squared(actual, predicted):
    correlation_matrix = np.corrcoef(actual, predicted)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    return r_squared

def calculate_mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    non_zero_actual = actual != 0
    return np.mean(np.abs((actual[non_zero_actual] - predicted[non_zero_actual]) / actual[non_zero_actual])) * 100


def calculate_mae(actual, predicted):
    return np.mean(np.abs(np.array(actual) - np.array(predicted)))

def calculate_mse(actual, predicted):
    return np.mean((np.array(actual) - np.array(predicted)) ** 2)

def calculate_me(actual, predicted):
    return np.mean(np.array(actual) - np.array(predicted))

def calculate_median_absolute_error(actual, predicted):
    return np.median(np.abs(np.array(actual) - np.array(predicted)))


def calculate_mase(actual, predicted, seasonal_period=1):
    actual, predicted = np.array(actual), np.array(predicted)
    n = len(actual)
    d = np.abs(np.diff(actual, n=seasonal_period)).sum() / (n - seasonal_period)
    errors = np.mean(np.abs(actual - predicted))
    return errors / d

def calculate_mbe(actual, predicted):
    return np.mean(np.array(actual) - np.array(predicted))

def calculate_time_series_values(Y, Sol, length):
    calculated_values = [0.0 for _ in range(length)]

    # Assuming the first value of Y is the initial value
    calculated_values[0] = Y[0]

    # Calculate the rest of the values based on the coefficients
    for t in range(1, length):
        value = Sol.a[0]  # This could be an intercept if your model has one
        for i in range(1, len(Sol.a)):
            # Adjust the index to be 0-based for accessing G
            value += Sol.a[i] * G[i-1](Y[t - 1])
        calculated_values[t] = value

    return calculated_values





def plot_time_series(original_data, calculated_data, series_number, save_path):
    plt.figure(figsize=(10, 6))

    # Create a new x-axis range that starts from 1
    x_axis_range = range(1, len(original_data) + 1)

    # Use the new x-axis range for plotting
    plt.plot(x_axis_range, original_data, label='Original Data', color='blue', linewidth=2)
    plt.plot(x_axis_range, calculated_data, label='GLDM Model', color='red', linestyle='dotted', linewidth=2)

    plt.title(f'Time Series: {case_name}: Original vs GLDM Model', fontsize=16)
    plt.xlabel('Time in years', fontsize=14)
    plt.ylabel('Production in tonnes', fontsize=14)

    # Automatically place the legend in the best location
    plt.legend(loc='best', fontsize=12)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save the plot to a file
    plt.savefig(save_path, format='png', dpi=300)

    # Optionally display the plot
    # plt.show()

    # Close the plot to free memory
    plt.close()



def plot_time_series_adjusted(original_data, calculated_data, series_number, save_path):
    plt.figure(figsize=(10, 6))

    # Ensure the first four values of calculated_data match the original_data
    if len(original_data) >= 2 and len(calculated_data) >= 2:
        for i in range(2):  # Update this loop to copy the first four values
            calculated_data[i] = original_data[i]

    # Adjust lengths if necessary to ensure both lists are of equal length
    min_length = min(len(original_data), len(calculated_data))

    # Modify here to remove the first and last value from both sets of data
    original_data_adjusted = original_data[1:min_length-1]  # Exclude the first and last items
    calculated_data_adjusted = calculated_data[1:min_length-1]  # Exclude the first and last items

    # Adjust the x-axis range to start from 1 after removing the first and last values
    # The new range needs to reflect the reduced number of data points
    x_axis_range = range(1, min_length - 1)  # Adjusted to start from 1 and match the new length
 
    # Plotting the original and calculated data
    plt.plot(x_axis_range, original_data_adjusted, label='Original Data', color='blue', linewidth=2)
    plt.plot(x_axis_range, calculated_data_adjusted, label='GLDM Model', color='red', linestyle='dotted', linewidth=2)

    # Setting the plot title and labels
    plt.title(f'Time Series: {case_name}: Original vs GLDM Model', fontsize=16)
    plt.xlabel('Time in years', fontsize=14)
    plt.ylabel('Production in tonnes', fontsize=14)

    # Adding a legend and grid
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Saving the plot to a file and closing the plot figure
    plt.savefig(save_path, format='png', dpi=300)
    plt.close()

    







def main():
    # Start measuring time and resources
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_memory_use = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

    # File handling
    with open("input.txt", "r") as f, open("output.txt", "w") as g:       
        # Data Input
        # Reading until ':' is encountered
        lc_c = ''
        while lc_c != ':':
            lc_c = f.read(1)
        m, ts = map(int, f.readline().split())
        global impl_len
        impl_len = m  # Length of time series
        global summs_count
        summs_count = 2  # Assuming summs_count is a known value
        print(f"Length: {m}\nTime series: {ts}\n")
        # End of Data Input

        
        
        # Reading time series data
        setnum = 0
        RY = [[] for _ in range(ts)]  # Create the ts arrays for time series
        for i in range(ts):
            RY[i] = [0.0] * (m + 2)
        while setnum < ts:
            print("Reading time series", setnum)
            ic = 1
            while ic <= m:
                line = f.readline()  # Read the next line
                s = float(line)
                RY[setnum][ic] = s
                ic += 1
            print("\n Finished reading of time series", setnum)
            setnum += 1
        GL_RY = RY
        # End Reading time series data

        
        
        # Writing results to a file
        g.write(f"Number of time series: {ts}\n")
        g.write(f"Length of time series: {impl_len}\n")      
        
        # Processing each time series
        for sn in range(ts):
            Y = [0.0] * (m + 2)
            k = 1
            for j in range(1, m + 1):
                Y[j] = RY[sn][j]
                k += 1
                # cout<<Y[j]<<" ";
            GL_Y = GL_RY[sn]

            lc_SST = np.zeros((summs_count + 1, summs_count * 2 + 2), dtype=float)
            # Solution
            GForming()
            print("GForming() OK\n")
            # Assuming lc_SST, GForming, SSTForming, JGTransforming, GLDMEstimator, and ForecastingEst functions are defined
            lc_SST = SSTForming(GL_Y)
            JGTransforming(summs_count, lc_SST)
            print("\n JGTransforming() OK\n")
            Sol = GLDMEstimator(GL_Y)
            print("GLDMEstimator() OK\n")


            # Calculate the time series values using the obtained coefficients
            calculated_ts_values = calculate_time_series_values(GL_Y, Sol, len(GL_Y))

            # Write the calculated time series values to the output file
            g.write("Calculated Time Series Values:\n")
            for val in calculated_ts_values:
                g.write(f"{val}\n")


            # Error calculations and table display
            original_data_trimmed = GL_Y  # Keep original data as is
            calculated_data_trimmed = calculated_ts_values [:]  # Ignore last two values from calculated data
            # Assuming the first value is manually set and should be excluded from error calculations
            start_index = 2 # Change to 2 if you need to skip the first two values for some reason

            # Ensure the lengths match
            min_length = min(len(original_data_trimmed), len(calculated_ts_values))
            original_data_trimmed = original_data_trimmed[:min_length]
            calculated_data_trimmed = calculated_ts_values[:min_length]

            # Calculate errors using consistent slicing
            mae = calculate_mae(original_data_trimmed[start_index:-1], calculated_data_trimmed[start_index:-1])
            mbe = calculate_mbe(original_data_trimmed[start_index:-1], calculated_data_trimmed[start_index:-1])
            mse = calculate_mse(original_data_trimmed[start_index:-1], calculated_data_trimmed[start_index:-1])
            rmse = calculate_rmse(original_data_trimmed[start_index:-1], calculated_data_trimmed[start_index:-1])
            r_squared = calculate_r_squared(original_data_trimmed[start_index:-1], calculated_data_trimmed[start_index:-1])
            mape = calculate_mape(original_data_trimmed[start_index:-1], calculated_data_trimmed[start_index:-1])
            me = calculate_me(original_data_trimmed[start_index:-1], calculated_data_trimmed[start_index:-1])
            median_abs_error = calculate_median_absolute_error(original_data_trimmed[start_index:-1], calculated_data_trimmed[start_index:-1])

            # Assuming a seasonal period for MASE calculation; adjust as necessary
            seasonal_period = 1  # This should be set based on your data's seasonality
            mase = calculate_mase(original_data_trimmed[start_index:-1], calculated_data_trimmed[start_index:-1], seasonal_period)
            g.write(f"Error Matrix start from second point to the end of dataset\n")
            # Write the results to the file
            g.write(f"RMSE: {rmse}\n")
            g.write(f"R-squared: {r_squared}\n")
            g.write(f"MAPE: {mape}\n")
            # Write the results to the file
            g.write(f"MAE: {mae}\n")
            g.write(f"MSE: {mse}\n")
            g.write(f"ME: {me}\n")
            g.write(f"Median Absolute Error: {median_abs_error}\n")
            # Write the results to the file
            g.write(f"MASE: {mase}\n")
            g.write(f"MBE: {mbe}\n")


            # Prepare and display the table
            g.write(f"{'Original Data':<20}{'Calculated Data':<20}{'Error':<20}\n")
            for orig, calc in zip(original_data_trimmed, calculated_data_trimmed):
                error = orig - calc
                g.write(f"{orig:<20}{calc:<20}{error:<20}\n")

            # Initialize ANS with 8 elements
            ANS = [0] * 8
            ANS[0] = sn  # Time Series Number
            ANS[1] = 0   # Placeholder for future use or additional data

            # Assigning model coefficients to ANS. Assuming Sol.a[0] is an intercept or similar.
            ANS[2] = Sol.a[1]  # Coefficient for G1
            ANS[3] = Sol.a[2]  # Coefficient for G2

            e = ForecastingEst(GL_Y, Sol)  # Forecasting Errors
            print("ForecastingEST OK\n")
            print(e.minFH, "\n", end='')
            ANS[4] = e.minFH  # Minimum Forecasting Horizon
            ANS[5] = e.D     # Average Absolute Error
            ANS[6] = e.E     # Average Error
            ANS[7] = Sol.Z   # Sum of Absolute Differences between Model and Actual Data

            # Writing the results with descriptive labels
            g.write(f"Time Series Number: {ANS[0]}\n")
            g.write("Model Coefficients:\n")
            g.write(f"Coefficient a1: {ANS[2]}\n")
            g.write(f"Coefficient a2: {ANS[3]}\n")
            g.write(f"Minimum Forecasting Horizon: {ANS[4]}\n")
            g.write(f"Average Absolute Error: {ANS[5]}\n")
            g.write(f"Average Error: {ANS[6]}\n")
            g.write(f"Sum of Absolute Differences: {ANS[7]}\n")

            # Using the first point of the time series for G function values
            x = Y[1]

            # Writing G function values
            g.write("G Function Values for a representative point (x):\n")
            g_val_1 = G[0](x)  # G1 function value
            g_val_2 = G[1](x)  # G2 function value
            g.write(f"G1 value: {g_val_1}\n")
            g.write(f"G2 value: {g_val_2}\n")
            g.write("Write original data:\n")
            # Write original data trimmed
            for value in original_data_trimmed[start_index:-1]:
                g.write(f"{value}\n")
            # Define column widths, adjust as needed based on expected data width
            col_widths = [25, 25, 25]

            # Write titles (headers) for each column with consistent spacing for a table-like display
            headers = ["Original Data", "Calculated Data", "Error"]
            header_line = "".join(f"{header:<{width}}" for header, width in zip(headers, col_widths))
            g.write(header_line + "\n")
            g.write("-" * sum(col_widths) + "\n")  # Divider line for visual separation

            # Assuming original_data_trimmed, calculated_data_trimmed, and start_index are defined
            # Calculate the length of data to be iterated over
            data_length = min(len(original_data_trimmed) - start_index - 1, len(calculated_data_trimmed) - start_index - 1)

            # Iterate over the range of data_length to access each element by its index
            for i in range(data_length):
                original = original_data_trimmed[start_index + i]  # Accessing the original data value
                calculated = calculated_data_trimmed[start_index + i]  # Accessing the calculated data value
                error = original - calculated  # Calculating the error between the original and calculated values
                
                # Writing the data and error to the file without rounding
                # Convert numbers to strings directly
                original_str = f"{original}"
                calculated_str = f"{calculated}"
                error_str = f"{error}"
                
                # Format the line with consistent spacing for a table-like display
                data_line = f"{original_str:<{col_widths[0]}}{calculated_str:<{col_widths[1]}}{error_str:<{col_widths[2]}}"
                g.write(data_line + "\n")

            original_data = Y[1:-1]  # Adjust indices as per your data
            calculated_data = calculate_time_series_values(GL_Y, Sol, len(GL_Y))[1:-1]

            # Define a path for saving the plot
            plot_save_path = f"plot_series_{sn}.png"  # This will save the plot in the current directory

            # Call the plotting function with the save path
            plot_time_series(original_data, calculated_data, sn, plot_save_path)
            # After calculating the calculated_ts_values
            # Define a path for saving the plot
            plot_save_path = f"plot_series_{sn}_direct_adjustment.png"  # Adjust the filename as needed

            # Call the updated plotting function with the save path
            plot_time_series_adjusted(original_data, calculated_ts_values, sn, plot_save_path)


            # Stop measuring time and resources
            end_time = time.time()
            final_memory_use = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

            # Calculate total time and memory used
            total_time = end_time - start_time
            total_memory_used = final_memory_use - initial_memory_use
            g.write("\nPerformance Metrics:\n")
            g.write(f"Total Execution Time: {total_time:.2f} seconds\n")
            g.write(f"Total Additional Memory Used: {total_memory_used:.2f} MB\n")
            
            


    # Wait for user input before closing (simulating 'Press any key' behavior)
    input("\nPress any key: ")

if __name__ == "__main__":
    main()
