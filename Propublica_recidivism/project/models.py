from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

class Models:
  def __init__(self, train_set, test_set):
    self.train_set, self.test_set = train_set, test_set
    self.X_train, self.y_train = train_set.features, train_set.labels.ravel()
    self.X_test, self.y_test = test_set.features, test_set.labels.ravel()


  def logistic_regression(self):
    logistic_regression = LogisticRegression(random_state = 0)
    scaler = StandardScaler()
    model = make_pipeline(scaler, logistic_regression)
    model.fit(self.X_train, self.y_train,
                          logisticregression__sample_weight = self.train_set.instance_weights)
    test_predictions = model.predict(self.X_test)
    test_metrics = {'Accuracy': accuracy_score(self.y_test, test_predictions),
                    'Recall': recall_score(self.y_test, test_predictions),
                    'Precision': precision_score(self.y_test, test_predictions),
                    'F1 Score': f1_score(self.y_test, test_predictions)}
    test_confusion_matrix_ = confusion_matrix(self.y_test, test_predictions)
    return model, test_metrics, test_confusion_matrix_




    
  def svm(self):
    svm = SVC(random_state = 0, kernel = 'linear')
    scaler = StandardScaler()
    model = make_pipeline(scaler, svm)
    model.fit(self.X_train, self.y_train,
                          svc__sample_weight = self.train_set.instance_weights)
    predictions = model.predict(self.X_test)
    metrics = {'Accuracy': accuracy_score(self.y_test, predictions),
    'Recall': recall_score(self.y_test, predictions),
    'Precision': precision_score(self.y_test, predictions),
    'F1 Score': f1_score(self.y_test, predictions),}
    confusion_matrix_ = confusion_matrix(self.y_test, predictions)
    return model, metrics, confusion_matrix_
  
  def random_forest(self):
    random_forest = RandomForestClassifier(random_state = 0, n_estimators = 100)
    scaler = StandardScaler()
    model = make_pipeline(scaler, random_forest)
    model.fit(self.X_train, self.y_train,
                          randomforestclassifier__sample_weight = self.train_set.instance_weights)
    predictions = model.predict(self.X_test)
    metrics = {'Accuracy': accuracy_score(self.y_test, predictions),
    'Recall': recall_score(self.y_test, predictions),
    'Precision': precision_score(self.y_test, predictions),
    'F1 Score': f1_score(self.y_test, predictions),}
    confusion_matrix_ = confusion_matrix(self.y_test, predictions)
    return model, metrics, confusion_matrix_
