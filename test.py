'''
NOTE:
    1. iPDP is implemented only for the Brute version of PDP.
    2. The ICE version using pdpNew.from_estimator(kind="individual") replaces all NaN values with 0.
       Thus, it does not work correctly for ICE.
    3. sim_type = "L1", "L2". Default is cosine distance.

'''

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from pdpNew import pdpNew
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

data = pd.read_csv("Data/cleaned_dataset.csv")

X = data.drop('Biopsy', axis=1)
y = data['Biopsy']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = XGBClassifier(
    max_depth=15,
    subsample=0.33,
    objective='binary:logistic',
    n_estimators=100,
    learning_rate = 0.01,
    early_stopping_rounds=12,
    eval_metric=["error", "logloss"],
)

model = classifier.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

train_score = classifier.score(X_train, y_train)
val_score = classifier.score(X_val, y_val)

print(f"XGBoost Train Score: {train_score}")
print(f"XGBoost Validation Score: {val_score}")
print("Training set size:", X_train.shape)
print("Validation set size:", X_val.shape)

for i in range(X.shape[1]):
    #disp1 = PartialDependenceDisplay.from_estimator(model, X, [i],method="brute")
    #disp2 = PartialDependenceDisplay.from_estimator(model, X, [i], method="brute",kind="individual")
    disp3 = pdpNew.from_estimator(model, X, [i],method="brute",threshold=0.8,sim_type="L1")

    #disp1.plot(pdp_lim={1: (-0.05, 0.6)})
    #plt.savefig("Output/PDP_"+str(i)+".png")
    #disp2.plot(pdp_lim={1: (-0.05, 0.6)})
    #plt.savefig("Output/ICE_"+str(i)+".png")
    disp3.plot(pdp_lim={1: (-0.05, 0.6)})
    plt.savefig("Output/iPDP_" + str(i) + ".png")
