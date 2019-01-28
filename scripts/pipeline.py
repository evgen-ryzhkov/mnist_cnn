from scripts.data import MNISTData
from scripts.model import DigitClassifier


#  --------------
# Step 1
# getting data
data_obj = MNISTData()
X_train, y_train, X_test, y_test = data_obj.get_train_and_test_data()


#  --------------
# Step 2
# training model
clf_o = DigitClassifier()
clf_model = clf_o.train_model(X_train, y_train)


#  --------------
# Step 3
# evaluating model with test data
clf_o.evaluate_model(clf_model, X_train, y_train)

# total training time on CPU - 251s
