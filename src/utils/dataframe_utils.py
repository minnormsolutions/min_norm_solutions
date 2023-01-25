import numpy as np
import pandas as pd

def mse_0( value, target ):
    output = ( value - target )**2

    return output

def mse( series, target_position ):
    target = series.iloc[target_position]
    output = series.iloc[:target_position].apply( mse_0, args = (target,) )
    output = output.mean()

    return output

def row_mse( dataframe, target_row, end_column ):
    temp = dataframe.iloc[:target_row, :end_column] - dataframe.iloc[target_row, :-1]
    temp = temp**2
    new_col = temp.mean(axis = 1)

    return new_col