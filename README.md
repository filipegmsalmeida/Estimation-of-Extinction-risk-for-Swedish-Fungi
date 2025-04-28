# Estimation-of-Extinction-risk-for-Swedish-Fungi

First, you can find data for fungal species in Sweden in the file alldata.csv. Each row represents one species, with environmental (e.g., temperature) and human-related features (e.g., human population density index). This data was retrieved using a custom Python library (Baggström et al., in prep.). The AOO and EOO were also calculated, and the extinction risk category for each species (when available) was extracted from the Swedish Red List.

Second, a script is provided that presents a deep neural network structure. In this script, you can test different architectures and evaluate them using various metrics, such as balanced accuracy, to guide model selection. You have a variety of outputs available to help you identify the best-performing model.

