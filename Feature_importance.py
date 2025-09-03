#region Importing modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import balanced_accuracy_score
from sklearn.inspection import permutation_importance
import csv
#endregion

#region Functions
def scale_variables_min_max(df):
    # Predefined columns with their respective min and max values
    scaling_params = {
        'bioclim_bio01_mean_min': {'min': -290, 'max': 320},
        'bioclim_bio01_mean_max': {'min': -290, 'max': 320},
        'bioclim_bio12_mean_min': {'min': 0, 'max': 11401},
        'bioclim_bio12_mean_max': {'min': 0, 'max': 11401},
        's2_NDVI_mean_min': {'min': -1, 'max': 1},
        's2_NDVI_mean_max': {'min': -1, 'max': 1},
        'hii_hii_mean_min': {'min': 5, 'max': 5000},
        'hii_hii_mean_max': {'min': 5, 'max': 5000},
        'hii_population_density_mean_min': {'min': 0, 'max': 1000},
        'hii_population_density_mean_max': {'min': 0, 'max': 1000},
        'hii_power_mean_min': {'min': 0, 'max': 1000},
        'hii_power_mean_max': {'min': 0, 'max': 1000},
        'hii_roads_mean_min': {'min': 0, 'max': 1000},
        'hii_roads_mean_max': {'min': 0, 'max': 1000},
        'dem_b1_mean_min': {'min': 0, 'max': 2100},
        'dem_b1_mean_max': {'min': 0, 'max': 2100},
        's2_NDVI_std_min': {'min': -1, 'max': 1},
        's2_NDVI_std_max': {'min': -1, 'max': 1},
        'hii_hii_std_min': {'min': 5, 'max': 5000},
        'hii_hii_std_max': {'min': 5, 'max': 5000},
        'dem_b1_std_min': {'min': 0, 'max': 2100},
        'dem_b1_std_max': {'min': 0, 'max': 2100},
        'biomass_min': {'min': 0, 'max': 1246},
        'biomass_max': {'min': 0, 'max': 1246},
        'treeheight_min': {'min': 0, 'max': 500},
        'treeheight_max': {'min': 0, 'max': 500},
        'soilmoisture_min': {'min': -0.5, 'max': 102},
        'soilmoisture_max': {'min': -0.5, 'max': 102},
        'peatmap_min': {'min': 0, 'max': 88},
        'peatmap_max': {'min': 0, 'max': 88},
        #'groundcover_min': {'min': 0, 'max': 128},
        #'groundcover_max': {'min': 0, 'max': 128},
        'AOO2km': {'min': df['AOO2km'].min(), 'max': df['AOO2km'].max()},
        'EOOkm2': {'min': df['EOOkm2'].min(), 'max': df['EOOkm2'].max()},
        'n_occ': {'min': df['n_occ'].min(), 'max': df['n_occ'].max()}
    }

    for column, params in scaling_params.items():
        min_value = params['min']
        max_value = params['max']
        df[column] = (df[column] - min_value) / (max_value - min_value)

    return df

@tf.keras.utils.register_keras_serializable()
def balanced_accuracy(y_true, y_pred):
    y_true = tf.cast(tf.argmax(y_true, axis=1), tf.int32)
    y_pred = tf.cast(tf.argmax(y_pred, axis=1), tf.int32)

    cm = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32)
    tp = tf.linalg.diag_part(cm)
    fn = tf.reduce_sum(cm, axis=1) - tp

    recall_per_class = tp / (tp + fn + tf.keras.backend.epsilon())
    balanced_accuracy = tf.reduce_mean(recall_per_class)

    return balanced_accuracy

class BalancedAccuracyMetric(tf.keras.metrics.Metric):
    def __init__(self, num_classes=5, name='balanced_accuracy', **kwargs):
        super(BalancedAccuracyMetric, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros')
        self.fn = self.add_weight(name='fn', shape=(num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.argmax(y_true, axis=-1), tf.int32)  # Convert one-hot to class indices
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)  # Convert predictions to class indices

        # Compute confusion matrix
        cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32)

        # Compute True Positives (TP) and False Negatives (FN)
        tp = tf.linalg.diag_part(cm)
        fn = tf.reduce_sum(cm, axis=1) - tp

        # Update TP and FN
        self.tp.assign_add(tp)
        self.fn.assign_add(fn)

    def result(self):
        # Compute recalls per class
        recall_per_class = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        # Compute balanced accuracy
        balanced_accuracy = tf.reduce_mean(recall_per_class)
        return balanced_accuracy

    def reset_states(self):
        # Reset TP and FN
        self.tp.assign(tf.zeros((self.num_classes,), dtype=tf.float32))
        self.fn.assign(tf.zeros((self.num_classes,), dtype=tf.float32))

def balanced_accuracy_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    if y_pred.shape[1] > 1:  # Check if predictions are one-hot encoded
        y_pred = np.argmax(y_pred, axis=1)  # Convert to class indices
    return balanced_accuracy_score(y, y_pred)

def compute_permutation_importance(model, X_test, y_test, feature_names):
    # Compute permutation importance
    result = permutation_importance(
        model, X_test, y_test,
        scoring=balanced_accuracy_scorer,
        n_repeats=30,
        random_state=42
    )
    # Return results along with feature names
    return result.importances_mean, result.importances_std

def renaming_features(input_data):
    # Shuffle the data
    input_data = input_data.sample(frac=1, random_state=42).reset_index(drop=True)

    #Defining features and labels
    features = input_data.values[:,1:-1].astype(float)
    feature_names = input_data.columns[1:-1]
    labels = input_data.values[:,-1]

    # Dictionary for renaming
    rename_dict = {
        's2_NDVI_mean_min': 'NDVI_min',
        's2_NDVI_mean_max': 'NDVI_max',
        'hii_hii_mean_min': 'HII_min',
        'hii_hii_mean_max': 'HII_max',
        'hii_population_density_mean_min': 'HII_pop_min',
        'hii_population_density_mean_max': 'HII_pop_max',
        'hii_power_mean_min': 'HII_power_min',
        'hii_power_mean_max': 'HII_power_max',
        'hii_roads_mean_min': 'HII_roads_min',
        'hii_roads_mean_max': 'HII_roads_max',
        'dem_b1_mean_min': 'Dem_min',
        'dem_b1_mean_max': 'Dem_max',
        'bioclim_bio01_mean_min': 'Bio01_min',
        'bioclim_bio01_mean_max': 'Bio01_max',
        'bioclim_bio12_mean_min': 'Bio12_min',
        'bioclim_bio12_mean_max': 'Bio12_max',
        's2_NDVI_std_min': 'NDVI_std_min',
        's2_NDVI_std_max': 'NDVI_std_max',
        'hii_hii_std_min': 'HII_std_min',
        'hii_hii_std_max': 'HII_std_max',
        'dem_b1_std_min': 'Dem_std_min',
        'dem_b1_std_max': 'Dem_std_max'
    }

    # Convert the index to a Series
    index_series = pd.Series(feature_names)

    # Rename the values in the Series
    index_series = index_series.replace(rename_dict)

    # Convert back to an Index if needed
    feature_names = pd.Index(index_series)

    unique_values, counts = np.unique(labels, return_counts=True)
    # Display the results
    for value, count in zip(unique_values, counts):
        print(f"{value}: {count}")

    return feature_names


def calculating_featuring_importance(MODEL_PATH, model_name, num_folds, feature_names, OUTPUT_DIR):
    # Initialize lists to store importance scores
    all_importances_mean = []
    all_importances_std = []

    # Loop through each fold
    for fold in range(1, num_folds + 1):
        # Define paths for the current fold
        y_test_path = os.path.join(MODEL_PATH, f'y_test_{model_name}_fold{fold}.csv')
        X_test_path = os.path.join(MODEL_PATH,f'X_test_{model_name}_fold{fold}.csv')
        model_path = os.path.join(MODEL_PATH,f'model_{model_name}_fold{fold}.keras')  # Path to the saved model for the current fold

        # Debug prints to verify paths
        print(f"y_test_path: {y_test_path}")
        print(f"X_test_path: {X_test_path}")
        print(f"model_path: {model_path}")

        # Check if files exist before proceeding
        if not (os.path.exists(y_test_path) and os.path.exists(X_test_path) and os.path.exists(model_path)):
            print(f"One or more files do not exist for fold {fold}. Skipping...")
            continue

        # Load y_test and X_test for the current fold
        y_test = pd.read_csv(y_test_path)
        X_test = pd.read_csv(X_test_path).values

        if y_test.shape[1] > 1:
            y_test = np.argmax(y_test, axis=1)

        # Load the model for the current fold
        model = tf.keras.models.load_model(model_path,
                                           custom_objects={'BalancedAccuracyMetric': BalancedAccuracyMetric})

        # Compute permutation importance for the current fold
        importances_mean, importances_std = compute_permutation_importance(model, X_test, y_test, feature_names)

        # Store the results
        all_importances_mean.append(importances_mean)
        all_importances_std.append(importances_std)

    # Convert lists to numpy arrays
    all_importances_mean = np.array(all_importances_mean)
    all_importances_std = np.array(all_importances_std)

    # Compute the mean and standard deviation of importances across all folds
    mean_importances = np.mean(all_importances_mean, axis=0)
    std_importances = np.mean(all_importances_std, axis=0)

    # Sort features by importance
    sorted_idx = mean_importances.argsort()[::-1]

    # Prepare a list with feature names and their average importance values
    feature_importance_summary = []
    for idx in sorted_idx:
        feature_importance_summary.append((feature_names[idx], mean_importances[idx], std_importances[idx]))

    # Print feature importances
    print("Feature Importances (name, mean importance, std deviation):")
    for feature, mean_imp, std_imp in feature_importance_summary:
        print(f"{feature}: Mean Importance = {mean_imp:.4f}, Std Dev = {std_imp:.4f}")

    # Save feature importance summary to a CSV file
    csv_file = os.path.join(OUTPUT_DIR, f'feature_importance_summary_{model_name}.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Feature Name', 'Mean Importance', 'Std Deviation'])
        writer.writerows(feature_importance_summary)

    return mean_importances, std_importances, sorted_idx


def plotting_feature_importance(mean_importances, std_importances, sorted_idx, feature_names, model_name, OUTPUT_DIR):
    # Plot permutation importances
    fig_width = 22  # Width of the figure
    fig_height = 15  # Height of the figure
    label_font_size = 16  # Font size of the axis labels
    axis_font_size = 14  # Font size of the numbers on the x-axis
    title_font_size = 16  # Font size of the title
    padding = 0.1  # Padding between the labels

    plt.figure(figsize=(fig_width, fig_height))
    plt.barh(np.array(feature_names)[sorted_idx], mean_importances[sorted_idx], xerr=std_importances[sorted_idx],
             align='center')
    plt.xlabel('Permutation Importance', fontsize=label_font_size)
    plt.title('Feature Importance Averaged Across Folds', fontsize=title_font_size)

    # Increase the spacing between the labels
    plt.yticks(np.arange(len(feature_names)), np.array(feature_names)[sorted_idx], fontsize=label_font_size)

    # Increase the font size of the numbers on the x-axis
    plt.xticks(fontsize=axis_font_size)

    plt.subplots_adjust(left=padding, right=1 - padding, top=1 - padding, bottom=padding)

    plt.savefig(os.path.join(OUTPUT_DIR, f'feature_importance_{model_name}.png'), dpi=300,
        bbox_inches='tight')

    plt.show()
#endregion

def main(input_data, model_name, num_folds, OUTPUT_DIR):
    feature_names = renaming_features(input_data)
    mean_importances, std_importance, sorted_idx = calculating_featuring_importance(MODEL_PATH, model_name, num_folds, feature_names, OUTPUT_DIR)
    plotting_feature_importance(mean_importances, std_importance, sorted_idx, feature_names, model_name, OUTPUT_DIR)

if __name__ == "__main__":
    INPUT_PATH = 'data/all_data.csv'
    input_data = pd.read_csv(INPUT_PATH)

    MODEL_PATH = '/Users/filipegmsalmeida/Library/CloudStorage/OneDrive-Personal/Area_de_Trabalho/Projetos/Em andamento/Conservation Status Fungi Sweden/Submission/Scripts_final/results/models_and_test_data'

    OUTPUT_DIR = 'results/feature_importance'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Making sure that DD are excluded
    input_data['RedListCategory'] = input_data['RedListCategory'].replace('DD', np.nan)
    # Scaling all features between 0 and 1 (min and max)
    input_data = scale_variables_min_max(input_data)

    # Separating species with and without RL category
    RL_na = input_data[input_data['RedListCategory'].isna()]
    input_data = input_data.dropna()

    #Setting parameters
    num_folds = 5
    model_name = 'Oversampled_2_balanced_60_30_0.1_40'  # Oversampled_2_balanced_30_15_0.1_40  ----- Oversampled_5_balanced_30_15_8_0.1_40

    main(input_data, model_name, num_folds, OUTPUT_DIR)
