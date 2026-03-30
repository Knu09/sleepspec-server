The Flask server is responsible for hosting the SVM-based model used for mild sleep deprivation detection by the SleepSpec mobile application. It handles incoming audio data, processes it, and returns the model’s predictions.

### 📦 IMPORTANT
inside *updated_model/svm_pca_strf_n_comp=24.pkl* is a dictionary:
  "svm": clf.best_estimator_,
  "pca": pca
  
### TO RUN THE SERVER
Enter this command:
    `flask --app server run --host=0.0.0.0`
