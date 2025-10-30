from pygam import s, f, LinearGAM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('target', axis=1).to_numpy(),
    df['target'].to_numpy(),
    test_size=0.2,
    random_state=42,
)

# LinearGAM class for implementing Generalized Additive Models
gam = LinearGAM(n_splines=35)
print(gam)

gam.gridsearch(X_train, y_train)

print(gam.summary())

feature_names = diabetes.feature_names

grid_locs = [(0, 0), (0, 1), (1, 0), (1, 1)]

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

for i, feature in enumerate(feature_names[:4]):
    gi = grid_locs[i]
    XX = gam.generate_X_grid(term=i, n=100)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
    axs[gi].plot(XX[:, i], pdep, color='red', linewidth=2)
    axs[gi].fill_between(XX[:, i], confi[:, 0], confi[:, 1], color='red', alpha=0.2)
    axs[gi].set_xlabel(feature)
    axs[gi].set_ylabel('Partial dependence')
    axs[gi].set_title(f'Partial dependence of {feature}')
    axs[gi].grid(True)

plt.tight_layout()
plt.show()