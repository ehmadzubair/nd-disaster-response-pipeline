import numpy as np
import pandas as pd
from sqlalchemy import create_engine

messages = pd.read_csv('./messages.csv')
messages.head()