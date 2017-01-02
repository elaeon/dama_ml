import pymc as pm
import pandas as pd
import matplotlib.pyplot as plt


BINS = 30
df = pd.read_csv("/home/sc/test_data/numerai_datasets/numerai_training_data.csv")
results = df[df["target"] == 1]["feature1"].as_matrix()
avg = len(results) / BINS
plt.plot([0, 1], [avg, avg], 'r--')
plt.hist(results, bins=BINS)
plt.show()
