"""
Source code for the following paper:
Quintas-Martínez, V., Bahadori, M. T., Santiago, E., Mu, J. and Heckerman, D. 
"Multiply-Robust Causal Change Attribution" 
Proceedings of the 41st International Conference on Machine Learning, Vienna, Austria. PMLR 235, 2024.
"""

import warnings
from itertools import groupby
import numpy as np
from sklearn.base import is_classifier, is_regressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.pipeline import make_pipeline

class ThetaC:
    """Implements three estimators (regression, re-weighting, MR) for causal change attribution."""

    def __init__(self, C, h_fn=lambda y: y, warn_thr=1e-3):
        """
        Inputs:
        C = the change vector (a K+1 list of 0s and 1s).
        h_fn = the functional of interest. By default, the mean of y.
        """

        if any(x not in (0, 1) for x in C):
            raise ValueError(f"C must be a vector of 0s and 1s.")

        self.C = C
        self.h_fn = h_fn
        self.reg_dict = {}  # A dictionary to store the trained regressors
        self.cla_dict = {}  # A dictionary to store the trained classifiers
        self.calib_dict = {}  # A dictionary to store the trained calibrators
        self.alpha_dict = {} # A dictionary to store the fitted weights alpha_k (Theorem 2.4)
        self.warn_thr = warn_thr  # The threshold that generates warning about reweighting

    def _simplify_C(self, all_indep=False):
        """
        This function applies some simplifications to the change vector C,
        discussed in Appendix C of the paper.

        It creates:
        self.K_simpl: the number of groups after simplification (excluding the group that contains the outcome Y).
        self.C_simpl: the simplified change vector (a list of tuple. The first element in each tuple is a 0 or 1,
                      corresponding to the distribution we want to fix for the variables in that group. The second
                      element is a list containing the indices of the variables in that group).

        Inputs:
        all_indep = boolean, True if all explanatory variables are independent.
        """

        # When all variables are independent (Example C.4), simplify to a group of 0s and a group of 1s (regardless of order).
        if all_indep:
            unique = np.unique(self.C)
            self.C_simpl = sorted(
                [(c, [i for i in range(len(self.C)) if self.C[i] == c]) for c in unique],
                key=lambda a: np.max(a[1]),
            )

        # Otherwise, we just group the consecutive values (Remark C.1).
        else:
            self.C_simpl = [
                (c, list(inds)) for c, inds in groupby(range(len(self.C)), lambda i: self.C[i])
            ]

        self.K_simpl = len(self.C_simpl)-1

    def _train_reg(
        self,
        X_train,
        y_train,
        T_train,
        w_train=None,
        regressor=LinearRegression,
        regressor_args=(),
        regressor_kwargs=None,
        regressor_fit_kwargs=None,
        is_pipeline_reg=False,
    ):
        """
        This function trains the nested regression estimators, that will be stored in self.reg_dict.

        Inputs:
        X_train = (n_train, K) np.array with the X data (explanatory variables) for the training set.
        y_train = (n_train,) np.array with the Y data (outcome) for the training set.
        T_train = (n_train,) np.array with the T data (sample indicator) for the training set.
        w_train = optional (n_train,) np.array with sample weights for the train data.

        regressor = the regression estimator: a class supporting .fit and .predict methods.
        regressor_args = a tuple of positional args for regressor.__init__.
        regressor_kwargs = a dictionary of keyword args for regressor.__init__.
        regressor_fit_kwargs = a dictionary of keyword args for regressor.fit.
        is_pipeline_reg = True if regressor is a list of classes to be put into a Pipeline.
        """

        regressor_kwargs = {} if regressor_kwargs is None else regressor_kwargs
        regressor_fit_kwargs = {} if regressor_fit_kwargs is None else regressor_fit_kwargs

        # Train gamma_K:
        ind = T_train == self.C_simpl[-1][0]  # Select sample C_{K+1} \in {0,1}
        var = [a for b in self.C_simpl[:-1] for a in b[1]]  # Select right variables
        if not is_pipeline_reg:
            self.reg_dict[self.K_simpl-1] = regressor(*regressor_args, **regressor_kwargs)
        else:
            self.reg_dict[self.K_simpl-1] = make_pipeline(*[mm(*aa, **kk) for (mm, aa, kk) in zip(regressor, regressor_args, regressor_kwargs)])
        
        if w_train is not None:
            self.reg_dict[self.K_simpl-1].fit(
                X_train[np.ix_(ind, var)], self.h_fn(y_train[ind]), sample_weight = w_train[ind],
                **regressor_fit_kwargs)
        else:
            self.reg_dict[self.K_simpl-1].fit(
                X_train[np.ix_(ind, var)], self.h_fn(y_train[ind]), **regressor_fit_kwargs)

        # Train gamma_k for k = K-1, K-2, ..., 1:
        for k in range(2, self.K_simpl+1):
            ind = T_train == self.C_simpl[-k][0]  # Select sample C_{k+1} \in {0,1}
            var_new = [a for b in self.C_simpl[:-k] for a in b[1]]  # Select right variables
            # Use the fitted values from previous regression
            new_y = self.reg_dict[self.K_simpl-k+1].predict(X_train[np.ix_(ind, var)])
            if not is_pipeline_reg:
                self.reg_dict[self.K_simpl-k] = regressor(*regressor_args, **regressor_kwargs)
            else:
                self.reg_dict[self.K_simpl-k] = make_pipeline(*[mm(*aa, **kk) for (mm, aa, kk) in zip(regressor, regressor_args, regressor_kwargs)])
            if w_train is not None:
                self.reg_dict[self.K_simpl-k].fit(X_train[np.ix_(ind, var_new)], new_y, sample_weight = w_train[ind],
                                                **regressor_fit_kwargs)
            else:
                self.reg_dict[self.K_simpl-k].fit(X_train[np.ix_(ind, var_new)], new_y,
                                                **regressor_fit_kwargs)
            var = var_new

    def _train_cla(
        self,
        X_train,
        T_train,
        X_eval,
        w_train=None,
        classifier=LogisticRegression,
        classifier_args=(),
        classifier_kwargs=None,
        classifier_fit_kwargs=None,
        is_pipeline_cla=False,
        calibrator=None,
        X_calib=None,
        T_calib=None,
        w_calib=None,
        calibrator_args=(),
        calibrator_kwargs=None,
        calibrator_fit_kwargs=None,
    ):
        """
        This function trains the classification estimators for the weights, that will be stored in self.cla_dict.
        If calibrator is not None, it also calibrates the probabilities on a calibration set.

        Inputs:
        X_train = (n_train, K) np.array with the X data (explanatory variables) for the training set.
        T_train = (n_train,) np.array with the T data (sample indicator) for the training set.
        X_eval = (n_eval, K) np.array with the X data (explanatory variables) for the evaluation set.
                 Used only to give a warning about low overlap.
        w_train = optional (n_train,) np.array with sample weights for the training set.

        classifier = the classification estimator: a class supporting .fit and .predict_proba methods.
        classifier_args = a tuple of positional args for classifier.__init__.
        classifier_kwargs = a dictionary of keyword args for classifier.__init__.
        classifier_fit_kwargs = a dictionary of keyword args for classifier.fit.
        is_pipeline_cla = True if classifier is a list of classes to be put into a Pipeline.

        calibrator = Optional, a method for probability calibration on a calibration set.
                     This could be a regressor (e.g. sklearn.isotonic.IsotonicRegression) or
                     a classifier (e.g. sklearn.LogisticRegression).
                     No need to do this if classifier is a sklearn.calibration.CalibratedClassifierCV learner.
        X_calib = (n_calib, K) np.array with the X data (explanatory variables) for the calibration set.
        T_calib = (n_calib,) np.array with the T data (sample indicator) for the calibration set.
        w_calib = optional (n_calib,) np.array with sample weights for the calibration set.
        calibrator_args = a tuple of positional args for calibrator.__init__.
        calibrator_kwargs = a dictionary of keyword args for calibrator.__init__.
        calibrator_fit_kwargs = a dictionary of keyword args for calibrator.fit.
        """

        classifier_kwargs = {"penalty": None} if classifier_kwargs is None else classifier_kwargs
        classifier_fit_kwargs = {} if classifier_fit_kwargs is None else classifier_fit_kwargs
        calibrator_args = {} if calibrator_args is None else calibrator_args
        calibrator_fit_kwargs = {} if calibrator_fit_kwargs is None else calibrator_fit_kwargs

        # Train classifiers that will go into alpha_k for k = 1, ..., K:
        for k in range(self.K_simpl):
            var = [a for b in self.C_simpl[:(k+1)] for a in b[1]]  # Select right variables
            if not is_pipeline_cla:
                self.cla_dict[k] = classifier(*classifier_args, **classifier_kwargs)
            else:
                self.cla_dict[k] = make_pipeline(*[mm(*aa, **kk) for (mm, aa, kk) in zip(classifier, classifier_args, classifier_kwargs)])
            if w_train is not None:
                self.cla_dict[k].fit(X_train[:, var], T_train, sample_weight = w_train, **classifier_fit_kwargs)
            else:
                self.cla_dict[k].fit(X_train[:, var], T_train, **classifier_fit_kwargs)

            # For the case where you want to calibrate on different data,
            # No need if classifier is CalibratedClassifierCv
            if calibrator is not None: 
                proba = self.cla_dict[k].predict_proba(X_calib[:, var])[:, [1]]
                self.calib_dict[k] = calibrator(*calibrator_args, **calibrator_kwargs)
                if w_train is not None:
                    self.calib_dict[k].fit(proba, T_calib, sample_weight = w_calib, **calibrator_fit_kwargs)
                else:
                    self.calib_dict[k].fit(proba, T_calib, **calibrator_fit_kwargs)

        var = [a for b in self.C_simpl[:-1] for a in b[1]]  # Select right variables
        p = self.cla_dict[self.K_simpl-1].predict_proba(X_eval[:, var])[:, 1]

        if np.min(p) < self.warn_thr or np.max(p) > 1-self.warn_thr:
            warnings.warn(
                f"min P(T = 1 | X) = {np.min(p) :.2f}, max P(T = 1 | X) = {np.max(p) :.2f}, indicating low overlap. \n"
                + "Consider increasing the regularization for the classifier or using method = 'regression'."
            )

    def _get_alphas(self, X_eval, T_eval, ratio, calibrator=None, crop=1e-3):
        """
        This helper function uses the classifiers (and, if appropriate, the probability calibrators)
        to compute the weights alpha_k (defined in Theorem 2.4 of the paper), 
        which are then stored in self.alpha_dict.

        Inputs:
        k = int from 0 to K_simpl-1.
        X_eval = (n_eval, K) np.array with the X data (explanatory variables) for the evaluation set.
        T_eval = (n_eval,) np.array with the T data (sample indicator) for the evaluation set.
        ratio = n1/n0, unless the classifier has been trained with class weights.
        calibrator = Optional, a method for probability calibration on a calibration set.
                     This could be a regressor (e.g. sklearn.isotonic.IsotonicRegression) or
                     a classifier (e.g. sklearn.LogisticRegression).
                     No need to do this if classifier is a sklearn.calibration.CalibratedClassifierCV learner.
                     Used only to check if it is None.
        crop = float, all predicted probabilities from the classifier will be cropped below at this lower bound,
               and above at 1-crop.

        Returns:
        alpha_k = (n0,) or (n1,) np.array of alpha_k weights for sample C_{k+1} \in {0,1}
        """

        for k in range(self.K_simpl):
            ind = T_eval == self.C_simpl[k+1][0]  # Select sample C_{k+1} \in {0,1}
            
            # k = 0 doesn't have parents, get the marginal RN derivative.
            if k == 0:
                var = self.C_simpl[0][1]  # Select right variables
                p = np.minimum(np.maximum(self.cla_dict[0].predict_proba(X_eval[np.ix_(ind, var)])[:, 1],
                                        crop), 1-crop)
                if calibrator is not None and is_regressor(calibrator):
                    p = np.minimum(np.maximum(self.calib_dict[0].predict(p[:, np.newaxis]), crop), 1-crop)
                elif calibrator is not None and is_classifier(calibrator):
                    p = np.minimum(np.maximum(self.calib_dict[0].predict_proba(p[:, np.newaxis])[:, 1],
                                            crop), 1-crop)
                w = (1.0-p)/p * ratio if self.C_simpl[k+1][0] else p/(1.0-p) * 1/ratio

            # For k > 0 get the conditional RN derivative dividing the RN derivative for \bar{X}_j
            # by the RN derivative for \bar{X}_{j-1}.
            else:
                var_joint = [a for b in self.C_simpl[:(k+1)] for a in b[1]] # Variables up to k
                p_joint = np.minimum(np.maximum(self.cla_dict[k].predict_proba(X_eval[np.ix_(ind, var_joint)])[:, 1],
                                                crop), 1-crop)
                if calibrator is not None and is_regressor(calibrator):
                    p_joint = np.minimum(np.maximum(self.calib_dict[k].predict(p_joint[:, np.newaxis]), 
                                                    crop), 1-crop)
                elif calibrator is not None and is_classifier(calibrator):
                    p_joint = np.minimum(np.maximum(self.calib_dict[k].predict_proba(p_joint[:, np.newaxis])[:, 1],
                                                    crop), 1-crop)
                w_joint = (1-p_joint)/p_joint if self.C_simpl[k+1][0] else p_joint/(1-p_joint)

                var_cond = [a for b in self.C_simpl[:k] for a in b[1]]  # Variables up to k-1
                p_cond = np.minimum(np.maximum(self.cla_dict[k-1].predict_proba(X_eval[np.ix_(ind, var_cond)])[:, 1],
                                            crop), 1-crop)
                if calibrator is not None and is_regressor(calibrator):
                    p_cond = np.minimum(np.maximum(self.calib_dict[k-1].predict(p_cond[:, np.newaxis]),
                                                crop), 1-crop)
                if calibrator is not None and is_classifier(calibrator):
                    p = np.minimum(np.maximum(self.calib_dict[k-1].predict_proba(p_cond[:, np.newaxis])[:, 1],
                                            crop), 1-crop)
                w_cond = p_cond/(1-p_cond) if self.C_simpl[k+1][0] else (1-p_cond)/p_cond

                w = w_joint * w_cond

            self.alpha_dict[k] = w * self.alpha_dict[k-2] if k-2 in self.alpha_dict.keys() else w

        # alpha_k should integrate to 1. In small samples this might not be the case, so we standardize:
        self.alpha_dict[k] /= np.mean(self.alpha_dict[k])

    def est_scores(
        self,
        X_eval,
        y_eval,
        T_eval,
        X_train,
        y_train,
        T_train,
        w_eval=None,
        w_train=None,
        method="MR",
        regressor=LinearRegression,
        regressor_args=(),
        regressor_kwargs=None,
        regressor_fit_kwargs=None,
        is_pipeline_reg=False,
        classifier=LogisticRegression,
        classifier_args=(),
        classifier_kwargs=None,
        classifier_fit_kwargs=None,
        is_pipeline_cla=False,
        calibrator=None,
        X_calib=None,
        T_calib=None,
        w_calib=None,
        calibrator_args=(),
        calibrator_kwargs=None,
        calibrator_fit_kwargs=None,
        all_indep=False,
        crop=1e-3,
    ):
        """
        This function computes the scores that are averaged to get each theta_hat.
        These are psi_hat in the notation of Section 2.5 of the paper.
        It is convenient to have a function that returns the scores,
        rather than just theta_hat, to compute things like bootstrapped standard errors.

        Inputs:
        X_eval = (n_eval, K) np.array with the X data (explanatory variables) for the evaluation set.
        y_eval = (n_eval,) np.array with the Y data (outcome) for the evaluation set.
        T_eval = (n_eval,) np.array with the T data (sample indicator) for the evaluation set.
        X_train = (n_train, K) np.array with the X data (explanatory variables) for the training set.
        y_train = (n_train,) np.array with the Y data (outcome) for the training set.
        T_train = (n_train,) np.array with the T data (sample indicator) for the training set.
        w_eval = optional (n_eval,) np.array with sample weights for the evaluation set.
        w_train = optional (n_train,) np.array with sample weights for the training set.

        method = One of 'regression', 're-weighting', 'MR'. By default, 'MR'.

        regressor = the regression estimator: a class supporting .fit and .predict methods.
        regressor_args = a tuple of positional args for regressor.__init__.
        regressor_kwargs = a dictionary of keyword args for regressor.__init__.
        regressor_fit_kwargs = a dictionary of keyword args for regressor.fit.
        is_pipeline_reg = True if regressor is a list of classes to be put into a Pipeline.
        
        classifier = the classification estimator: a class supporting .fit and .predict_proba methods.
        classifier_args = a tuple of positional args for classifier.__init__.
        classifier_kwargs = a dictionary of keyword args for classifier.__init__.
        classifier_fit_kwargs = a dictionary of keyword args for classifier.fit.
        is_pipeline_cla = True if classifier is a list of classes to be put into a Pipeline.
        
        calibrator = Optional, a method for probability calibration on a calibration set.
                     This could be a regressor (e.g. sklearn.isotonic.IsotonicRegression) or
                     a classifier (e.g. sklearn.LogisticRegression).
                     No need to do this if classifier is a sklearn.calibration.CalibratedClassifierCV learner.
        X_calib = (n_calib, K) np.array with the X data (explanatory variables) for the calibration set.
        T_calib = (n_calib,) np.array with the T data (sample indicator) for the calibration set.
        w_calib = optional (n_calib,) np.array with sample weights for the calibration set.
        calibrator_args = a tuple of positional args for calibrator.__init__.
        calibrator_kwargs = a dictionary of keyword args for calibrator.__init__.
        calibrator_fit_kwargs = a dictionary of keyword args for calibrator.fit.

        all_indep = boolean, True if all explanatory variables are independent (used for self._simplify_C).
        crop = float, all predicted probabilities from the classifier will be cropped below at this lower bound,
               and above at 1-crop.

        Returns:
        theta_scores = (n_eval,) np.array of scores, such that theta_hat = np.mean(theta_scores).
        """
        
        regressor_kwargs = {} if regressor_kwargs is None else regressor_kwargs
        regressor_fit_kwargs = {} if regressor_fit_kwargs is None else regressor_fit_kwargs
        classifier_kwargs = {"penalty": None} if classifier_kwargs is None else classifier_kwargs
        classifier_fit_kwargs = {} if classifier_fit_kwargs is None else classifier_fit_kwargs
        calibrator_args = {} if calibrator_args is None else calibrator_args
        calibrator_fit_kwargs = {} if calibrator_fit_kwargs is None else calibrator_fit_kwargs

        if w_eval is None:
            n0, n1, n = np.sum(1-T_eval), np.sum(T_eval), T_eval.shape[0]
        else:
            n0, n1, n = np.sum(w_eval*(1-T_eval)), np.sum(w_eval*T_eval), np.sum(w_eval)

        if len(self.C) != X_train.shape[1]+1:
            raise ValueError(f"len(C) must be K+1={X_train.shape[1]+1}, not {len(self.C)}")

        self._simplify_C(all_indep=all_indep)

        if self.K_simpl > 0:
            if method == "regression":
                self._train_reg(
                    X_train,
                    y_train,
                    T_train,
                    w_train=w_train,
                    regressor=regressor,
                    regressor_args=regressor_args,
                    regressor_kwargs=regressor_kwargs,
                    regressor_fit_kwargs=regressor_fit_kwargs,
                    is_pipeline_reg=is_pipeline_reg,
                )

                ind = T_eval == self.C_simpl[0][0]  # Select sample C_1 \in {0,1}
                var = self.C_simpl[0][1]  # Select right variables
                if self.C_simpl[0][0] == 1:
                    theta_scores = np.concatenate(
                        (
                            np.zeros_like(y_eval[T_eval==0]),
                            self.reg_dict[0].predict(X_eval[np.ix_(ind, var)])*n/n1,
                        )
                    )
                else:
                    theta_scores = np.concatenate(
                        (
                            self.reg_dict[0].predict(X_eval[np.ix_(ind, var)])*n/n0,
                            np.zeros_like(y_eval[T_eval==1]),
                        )
                    )

            elif method == "re-weighting":
                self._train_cla(
                    X_train,
                    T_train,
                    X_eval,
                    w_train=w_train,
                    classifier=classifier,
                    classifier_args=classifier_args,
                    classifier_kwargs=classifier_kwargs,
                    classifier_fit_kwargs=classifier_fit_kwargs,
                    is_pipeline_cla=is_pipeline_cla,
                    calibrator=calibrator,
                    X_calib=X_calib,
                    T_calib=T_calib,
                    w_calib=w_calib,
                    calibrator_args=calibrator_args,
                    calibrator_kwargs=calibrator_kwargs,
                    calibrator_fit_kwargs=calibrator_fit_kwargs,
                )

                if 'class_weight' not in classifier_kwargs:
                    ratio = n1/n0
                elif classifier_kwargs['class_weight'] == 'balanced':
                    ratio = 1
                else:
                    ratio = n1/n0 * classifier_kwargs['class_weight'][0]/classifier_kwargs['class_weight'][1]

                self._get_alphas(
                    X_eval, 
                    T_eval, 
                    ratio,
                    calibrator=calibrator, 
                    crop=crop
                )
                
                ind = T_eval == self.C_simpl[-1][0]  # Select sample C_{K+1} \in {0,1}
                if self.C_simpl[-1][0] == 1:
                    theta_scores = np.concatenate(
                        (np.zeros_like(y_eval[T_eval==0]), self.alpha_dict[self.K_simpl-1] * self.h_fn(y_eval[ind])*n/n1)
                    )
                else:
                    theta_scores = np.concatenate(
                        (self.alpha_dict[self.K_simpl-1] * self.h_fn(y_eval[ind])*n/n0, np.zeros_like(y_eval[T_eval==1]))
                    )

            elif method == "MR":
                theta_scores_0 = np.zeros_like(y_eval[T_eval==0])
                theta_scores_1 = np.zeros_like(y_eval[T_eval==1])

                # Regression base estimate:
                self._train_reg(
                    X_train,
                    y_train,
                    T_train,
                    w_train=w_train,
                    regressor=regressor,
                    regressor_args=regressor_args,
                    regressor_kwargs=regressor_kwargs,
                    regressor_fit_kwargs=regressor_fit_kwargs,
                    is_pipeline_reg=is_pipeline_reg,
                )

                ind = T_eval == self.C_simpl[0][0]  # Select sample C_1 \in {0,1}
                var = self.C_simpl[0][1]  # Select right variables
                if self.C_simpl[0][0] == 1:
                    theta_scores_1 += self.reg_dict[0].predict(X_eval[np.ix_(ind, var)])
                else:
                    theta_scores_0 += self.reg_dict[0].predict(X_eval[np.ix_(ind, var)])

                # Debiasing terms up to K-1:
                self._train_cla(
                    X_train,
                    T_train,
                    X_eval,
                    w_train=w_train,
                    classifier=classifier,
                    classifier_args=classifier_args,
                    classifier_kwargs=classifier_kwargs,
                    classifier_fit_kwargs=classifier_fit_kwargs,
                    is_pipeline_cla=is_pipeline_cla,
                    calibrator=calibrator,
                    X_calib=X_calib,
                    T_calib=T_calib,
                    w_calib=w_calib,
                    calibrator_args=calibrator_args,
                    calibrator_kwargs=calibrator_kwargs,
                    calibrator_fit_kwargs=calibrator_fit_kwargs,
                )

                if 'class_weight' not in classifier_kwargs:
                    ratio = n1/n0
                elif classifier_kwargs['class_weight'] == 'balanced':
                    ratio = 1
                else:
                    ratio = n1/n0 * classifier_kwargs['class_weight'][1]/classifier_kwargs['class_weight'][0]

                self._get_alphas(
                    X_eval, 
                    T_eval, 
                    ratio,
                    calibrator=calibrator, 
                    crop=crop
                )

                for k in range(self.K_simpl):
                    ind = T_eval == self.C_simpl[k+1][0]  # Select sample C_{k+1} \in {0,1}
                    var = [a for b in self.C_simpl[:(k+1)] for a in b[1]]  # Variables up to k
                    var_next = [a for b in self.C_simpl[:(k+2)] for a in b[1]]  # Variables up to k+1
                    if self.C_simpl[k+1][0] == 1:
                        if k < self.K_simpl-1:
                            theta_scores_1 += self.alpha_dict[k] * (
                                self.reg_dict[k+1].predict(X_eval[np.ix_(ind, var_next)])
                                - self.reg_dict[k].predict(X_eval[np.ix_(ind, var)])
                            )
                        else:
                            theta_scores_1 += self.alpha_dict[k] * (
                                self.h_fn(y_eval[ind])
                                - self.reg_dict[self.K_simpl-1].predict(X_eval[np.ix_(ind, var)])
                            )
                    else:
                        if k < self.K_simpl-1:
                            theta_scores_0 += self.alpha_dict[k] * (
                                self.reg_dict[k+1].predict(X_eval[np.ix_(ind, var_next)])
                                - self.reg_dict[k].predict(X_eval[np.ix_(ind, var)])
                            )
                        else:
                            theta_scores_0 += self.alpha_dict[k] * (
                                self.h_fn(y_eval[ind])
                                - self.reg_dict[self.K_simpl-1].predict(X_eval[np.ix_(ind, var)])
                            )

                theta_scores = np.concatenate((theta_scores_0*n/n0, theta_scores_1*n/n1))

            else:
                raise AttributeError(f'Method "{method}" Not Implemented')

        # When C = [1, 1, ..., 1] we can just take the sample mean of y_eval[T_eval == 1]
        elif self.C_simpl[0][0] == 1:
            theta_scores = np.concatenate((np.zeros_like(y_eval[T_eval==0]), self.h_fn(y_eval[T_eval == 1])*n/n1))

        # When C = [0, 0, ..., 0] we can just take the sample mean of y_eval[T_eval == 0]
        else:
            theta_scores = np.concatenate((self.h_fn(y_eval[T_eval == 0])*n/n0, np.zeros_like(y_eval[T_eval==1])))

        return theta_scores

    def est_theta(
        self,
        X_eval,
        y_eval,
        T_eval,
        X_train,
        y_train,
        T_train,
        w_eval=None,
        w_train=None,
        method="MR",  # One of 'regression', 're-weighting', 'MR',
        regressor=LinearRegression,
        regressor_args=(),
        regressor_kwargs=None,
        regressor_fit_kwargs=None,
        is_pipeline_reg=False,
        classifier=LogisticRegression,
        classifier_args=(),
        classifier_kwargs=None,
        classifier_fit_kwargs=None,
        is_pipeline_cla=False,
        calibrator=None,
        X_calib=None,
        T_calib=None,
        w_calib=None,
        calibrator_args=(),
        calibrator_kwargs=None,
        calibrator_fit_kwargs=None,
        all_indep=False,
        crop=1e-3,
    ):
        """
        This function computes the scores that are averaged to get each theta_hat,
        and then returns (theta_hat, std_error)

        Inputs:
        X_eval = (n_eval, K) np.array with the X data (explanatory variables) for the evaluation set.
        y_eval = (n_eval,) np.array with the Y data (outcome) for the evaluation set.
        T_eval = (n_eval,) np.array with the T data (sample indicator) for the evaluation set.
        X_train = (n_train, K) np.array with the X data (explanatory variables) for the training set.
        y_train = (n_train,) np.array with the Y data (outcome) for the training set.
        T_train = (n_train,) np.array with the T data (sample indicator) for the training set.
        w_eval = optional (n_eval,) np.array with sample weights for the evaluation set.
        w_train = optional (n_train,) np.array with sample weights for the training set.

        method = One of 'regression', 're-weighting', 'MR'. By default, 'MR'.

        regressor = the regression estimator: a class supporting .fit and .predict methods.
        regressor_args = a tuple of positional args for regressor.__init__.
        regressor_kwargs = a dictionary of keyword args for regressor.__init__.
        regressor_fit_kwargs = a dictionary of keyword args for regressor.fit.
        is_pipeline_reg = True if regressor is a list of classes to be put into a Pipeline.
        
        classifier = the classification estimator: a class supporting .fit and .predict_proba methods.
        classifier_args = a tuple of positional args for classifier.__init__.
        classifier_kwargs = a dictionary of keyword args for classifier.__init__.
        classifier_fit_kwargs = a dictionary of keyword args for classifier.fit.
        is_pipeline_cla = True if classifier is a list of classes to be put into a Pipeline.

        calibrator = Optional, a method for probability calibration on a calibration set.
                     This could be a regressor (e.g. sklearn.isotonic.IsotonicRegression) or
                     a classifier (e.g. sklearn.LogisticRegression).
                     No need to do this if classifier is a sklearn.calibration.CalibratedClassifierCV learner.
        X_calib = (n_calib, K) np.array with the X data (explanatory variables) for the calibration set.
        T_calib = (n_calib,) np.array with the T data (sample indicator) for the calibration set.
        w_train = optional (n_calib,) np.array with sample weights for the calibration set.
        calibrator_args = a tuple of positional args for calibrator.__init__.
        calibrator_kwargs = a dictionary of keyword args for calibrator.__init__.
        calibrator_fit_kwargs = a dictionary of keyword args for calibrator.fit.

        all_indep = boolean, True if all explanatory variables are independent (used for self._simplify_C).
        crop = float, all predicted probabilities from the classifier will be cropped below at this lower bound,
               and above at 1-crop.

        Returns:
        theta_hat = the point estimate, np.mean(theta_scores) for the scores computed by self.est_scores.
        std_err = the standard error for theta_hat, sem(theta_scores) for the scores computed by self.est_scores.
        """

        regressor_kwargs = {} if regressor_kwargs is None else regressor_kwargs
        regressor_fit_kwargs = {} if regressor_fit_kwargs is None else regressor_fit_kwargs
        classifier_kwargs = {"penalty": None} if classifier_kwargs is None else classifier_kwargs
        classifier_fit_kwargs = {} if classifier_fit_kwargs is None else classifier_fit_kwargs
        calibrator_args = {} if calibrator_args is None else calibrator_args
        calibrator_fit_kwargs = {} if calibrator_fit_kwargs is None else calibrator_fit_kwargs

        theta_scores = self.est_scores(
            X_eval,
            y_eval,
            T_eval,
            X_train,
            y_train,
            T_train,
            w_eval=w_eval,
            w_train=w_train,
            method=method,  # One of 'regression', 're-weighting', 'MR',
            regressor=regressor,
            regressor_args=regressor_args,
            regressor_kwargs=regressor_kwargs,
            regressor_fit_kwargs=regressor_fit_kwargs,
            is_pipeline_reg=is_pipeline_reg,
            classifier=classifier,
            classifier_args=classifier_args,
            classifier_kwargs=classifier_kwargs,
            classifier_fit_kwargs=classifier_fit_kwargs,
            is_pipeline_cla=is_pipeline_cla,
            calibrator=calibrator,
            X_calib=X_calib,
            T_calib=T_calib,
            w_calib=w_calib,
            calibrator_args=calibrator_args,
            calibrator_kwargs=calibrator_kwargs,
            calibrator_fit_kwargs=calibrator_fit_kwargs,
            all_indep=all_indep,
            crop=crop,
        )

        if w_eval is not None:
            w_sort = np.concatenate((w_eval[T_eval==0], w_eval[T_eval==1])) # Order weights in same way as scores
        else:
            w_sort = np.ones(np.shape(T_eval))
        weighted_stats = DescrStatsW(theta_scores, weights=w_sort, ddof=0)

        return weighted_stats.mean, weighted_stats.std_mean
