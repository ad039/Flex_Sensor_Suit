import pandas as pd
import numpy as np

xy = pd.read_csv('./PyTorch/Alex_Motion_Cap_1_edited.csv', skiprows=2, skip_blank_lines=0,na_values=[''])
xy = xy.dropna()
xy.drop("Frame", axis='columns')

xy.to_csv('./PyTorch/Alex_Motion_Cap_Formatted.csv')



