Flask server responsible for hosting the SVM-based sleep deprivation detection model used by the SleepSpec mobile application. It handles incoming audio data, processes it, and returns the model’s predictions in real-time.

inside *svm_with_pca_fold_4.pkl* is a dictionary:
  "svm": clf.best_estimator_,
  "pca": pca
