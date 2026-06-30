import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime as datetime

warnings.filterwarnings('ignore')

df = pd.read_csv("retail_sales.csv",parse_dates=["Date"])
