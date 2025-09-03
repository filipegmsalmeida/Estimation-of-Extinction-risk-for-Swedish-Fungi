# Estimation-of-Extinction-risk-for-Swedish-Fungi

First, you can find data for fungal species in Sweden in the file data/alldata.csv. Each row represents one species, with associated environmental (e.g., temperature) and human-related features (e.g., human population density index). These data were retrieved using a custom Python library (Baggström et al., in prep.). The AOO and EOO were also calculated, and the extinction risk category for each species (when available) was extracted from the Swedish Red List. A smaller subset of the data is also available, in case you only want to test scripts with a shorter run.

Using the R script in 01_correlation_plot you can check the correlation between the features included in the alldata.csv file. 

Using the Model_test_updated_balanced_acc script you can train a Neural Network model with different architectures and obtain outputs in a folder called results. There you will find:
  Saved models (save_model and models_and_test_data folders)
  Tables with summary statistics for the validation metric and the test dataset (tables folder)
  Plots showing the number of epochs used to train each model (train_plots folder)
  Confusion matrices for the best model (highest value in the metric chosen as the early-stop criterion for the validation dataset — confusion_matrix folder)

You will also find a custom script to plot all confusion matrices together (Plot Confusion Matrix). This script additionally converts the 5-class model into a 2-class model, generating a new CSV file with summary statistics and the corresponding confusion matrix plot.

In the Final_Model_Prediction script, you can train a specific model using 90% of the data for training and 10% for validation. You should use the best architecture and settings identified in the first script (see the tables folder and filter for the model with the highest value in the column mean_val_metric). In this script, you can also make predictions for species without any conservation status and then plot the results.

Finally, you can assess feature importance using the Feature_importance script. In addition, you can test whether there is bias related to the number of species occurrences by examining the relationship with AOO and EOO, and whether sampling density (number of occurrences / EOO or AOO) influences prediction accuracy (balanced accuracy).
