import sys
import os
import pandas as pd
from settings import Settings
from src.functions.functions_plot import example_plot

settings = Settings()

train = pd.read_csv(settings.config['Data Locations'].get('train'))





ax = example_plot(train['trip_duration'], output_file_path=settings.figures)
