
import numpy as np
import torch

from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import classification_report
from sklearn.svm import SVC

#from __future__ import print_function

class cross_validator:
    def __init__(self, training_x, training_y, testing_x, testing_y):
        """
        Note:
            Using sklearn's gridsearch allows us to generate the most optimal hyperparameters for 
                  our svm classifier. 
            Choosing a very general (and average) range of hyper parameters it runs through 
                  all the combinations and outputs the best combo given the data set.

        Args:
             training_x: Torch tensor
                  Contains spectra (flux units) with the same shapes to be trained on.
             training_y: Torch tensor
                  Holds corresponding labels. 
             testing_x: Torch tensor
                  Contains spectra (flux units) with the same shapes to be tested on.
             testing_y: Torch tensor 
                  Holds corresponding labels for the testing set. 
             

        Returns:
            parameter(1,2): float
                 Outputs the parameters specified. 
                 To Rows and columns represent 1 2 3 for classification.
                 For the x=y line (diagonal from bottom left to top right)
                      it indicates correct classification.
                 For all other ones it indicates incorrect classification

        """
        c = np.logspace(0, 4)
        gamma = [0.0001,0.001,0.01,.1,1,10,100,1000]
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': gamma,
                             'C': c}]

        claf = SVC()
        param = 'C'
        param1 = 'gamma'

        scores = ['precision', 'recall']
        loss_function=torch.nn.MSELoss(size_average=False)
        for score in scores:
            cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)

            clf = GridSearchCV(estimator = claf, param_grid = tuned_parameters, cv=cv,
                               scoring='%s_macro' % score)
            clf.fit(training_x, training_y)

            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            y_true, y_pred = torch.from_numpy(testing_y), torch.from_numpy(clf.predict(testing_x))

            self.parameter1 = clf.best_estimator_.get_params()[param]
            self.parameter2 = clf.best_estimator_.get_params()[param1]
            self.loss = loss_function(np.round(y_pred), y_true)

