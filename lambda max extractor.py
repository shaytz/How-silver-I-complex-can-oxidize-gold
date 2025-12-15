import sys
import pandas as pd
from scipy.optimize import curve_fit
from numpy import exp, linspace, random
import pylab as plb
import matplotlib.pyplot as plt

def gaussian(x, amp, cen, wid):
    return amp * exp(-(x-cen)**2 / wid)

def main():
    path = 'C:/Users/Medhanie/Desktop/Tzuf/crude_results/UV-VIS/UV-VIS measurements Cary/271125/seeds Au citrate+ K[Ag(CN)2].csv'
    raw_df = pd.read_csv(path,header=None, index_col=False)
    results_array = [['Experiment','Lambda Max','Absorbance']]
    start_wavelength = float(500)
    end_wavelength = float(550)
    column_index_raw_df = 0
    row_index_raw_df = 0

    while True:
        x_wavelength_array = []
        y_absorbance_array = []
        if column_index_raw_df >= len(raw_df.columns):
            break
        experiment = raw_df.iat[row_index_raw_df,column_index_raw_df]
        if experiment == 'NaN':
            break
        lambda_max = 0
        max_absorbance = 0
        row_index_raw_df += 2
        while True:
            if row_index_raw_df == len(raw_df):
                break
            if  raw_df.iat[row_index_raw_df,column_index_raw_df] == 'NaN':
                break
            try:
                wavelength = float(raw_df.iat[row_index_raw_df,column_index_raw_df])
            except:
                break
            if (wavelength <= end_wavelength) and (wavelength >= start_wavelength):
                Abs = float(raw_df.iat[row_index_raw_df, column_index_raw_df + 1])
                x_wavelength_array.append(wavelength)
                y_absorbance_array.append(Abs)
                if Abs > max_absorbance:
                    max_absorbance = Abs
                    lambda_max = wavelength
            row_index_raw_df += 1
        if x_wavelength_array == []:
            break
        init_vals = [max_absorbance, lambda_max, 1]  # for [amp, cen, wid]
        try:
            best_vals, covar = curve_fit(gaussian, x_wavelength_array, y_absorbance_array, p0=init_vals)
        except:
            results_array.append([experiment, "-", "-"])
            row_index_raw_df = 0
            column_index_raw_df += 2
            continue
        print(best_vals)
        plt.plot(x_wavelength_array, y_absorbance_array, 'b+:', label='data')
        plt.plot(x_wavelength_array, gaussian(x_wavelength_array, *best_vals), 'ro:', label='fit')
        plt.legend()
        plt.title('Fig. 3 - Fit for Time Constant')
        plt.xlabel('wavelength')
        plt.ylabel('Abs')
        plt.show()
        results_array.append([experiment, best_vals[1], best_vals[0]])
        row_index_raw_df = 0
        column_index_raw_df += 2
    df = pd.DataFrame(results_array)
    new_path = path[:-4] + "_results" + '.xlsx'
    df.to_excel(new_path, index=False)
    print('finished:)')


main()
