#region Importing modules
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
#endregion

#region Function
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


class BalancedAccuracyEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_balanced_accuracy', patience=5, verbose=1, restore_best_weights=False):
        super(BalancedAccuracyEarlyStopping, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.best = -float('inf')
        self.best_weights = None
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if current > self.best:
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                if self.verbose > 0:
                    print(f'\nEpoch {epoch + 1}: early stopping with best {self.monitor} = {self.best:.4f}')

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f'\nRestoring model weights from the end of the best epoch: {self.stopped_epoch + 1}')

def plot_training_history(history, title, save_path, model_type='classification', show_best_epoch=True, show=True):
    fig = plt.figure(figsize=(8, 5))

    if model_type == 'classification':
        plt.plot(history.history['balanced_accuracy'], label='Training set Balanced Acc')
        plt.plot(history.history['val_balanced_accuracy'], label='Validation set Balanced Acc')
        plt.ylabel('Balanced_acc')

        if show_best_epoch:
            best_epoch = np.argmax(history.history['val_balanced_accuracy'])
            plt.axvline(best_epoch, c='grey', linestyle='--')
            plt.axhline(history.history['val_balanced_accuracy'][best_epoch], c='grey', linestyle='--')
            plt.gca().axvspan(best_epoch, len(history.history['balanced_accuracy']), color='grey', alpha=0.3, zorder=3)

    elif model_type == 'regression':
        plt.plot(history.history['balanced_accuracy'], label='Training set MAE')
        plt.plot(history.history['val_balanced_accuracy'], label='Validation set MAE')
        plt.ylabel('Mean Absolute Error (MAE)')

        if show_best_epoch:
            best_epoch = np.argmin(history.history['val_balanced_accuracy'])
            plt.axvline(best_epoch, c='grey', linestyle='--')
            plt.axhline(history.history['val_balanced_accuracy'][best_epoch], c='grey', linestyle='--')
            plt.gca().axvspan(best_epoch, len(history.history['balanced_accuracy']), color='grey', alpha=0.3, zorder=3)

    plt.xlabel('Epochs')
    plt.title('\n'.join(title.split()), fontsize=12)
    plt.grid()
    plt.legend(loc='upper center')

    if show == True:
        plt.show()

    fig.savefig(save_path)
    plt.close(fig)


def training_model(input_data, data_to_predict, output_path, batch_size, Dropout, hidden_layers, num_classes, oversampling, weight):
    features = input_data.values[:, 1:-1].astype(float)
    feature_names = input_data.columns[1:-1]
    labels = input_data.values[:, -1]

    # Defining features for prediciton
    features_to_predict = data_to_predict.values[:, 1:-1].astype(float)
    feature_names_to_predict = data_to_predict.columns[1:-1]

    if num_classes == 2:
        mapping = {'LC': 'NEN', 'NT': 'NEN', 'VU': 'EN', 'EN': 'EN', 'CR': 'EN'}
        labels = np.array([mapping[item] for item in labels])

    unique_values, counts = np.unique(labels, return_counts=True)
    # Display the results
    for value, count in zip(unique_values, counts):
        print(f"{value}: {count}")

    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.1, random_state=42,
                                                      stratify=labels)

    if oversampling == 'yes':
        smote = SMOTE(k_neighbors=5, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    if num_classes == 5:
        mapping = {'LC': 0, 'NT': 1, 'VU': 2, 'EN': 3, 'CR': 4}
        # Translate the array using the mapping
        y_train = np.array([mapping[item] for item in y_train])
        y_train = tf.keras.utils.to_categorical(y_train)
        y_val = np.array([mapping[item] for item in y_val])
        y_val = tf.keras.utils.to_categorical(y_val)


    if num_classes == 2:
        mapping = {'NEN': 0, 'EN': 1}
        # Translate the array using the mapping
        y_train = np.array([mapping[item] for item in y_train])
        y_train = tf.keras.utils.to_categorical(y_train)
        y_val = np.array([mapping[item] for item in y_val])
        y_val = tf.keras.utils.to_categorical(y_val)

    class_weights = None
    if weight == 'yes' and oversampling == 'no':
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(np.argmax(y_train, axis=1)),
            y=np.argmax(y_train, axis=1)
        )
        class_weights = dict(enumerate(class_weights))

    # region Setting Architecture
    architecture = []
    # Input layer
    architecture.append(tf.keras.layers.Flatten(input_shape=[X_train.shape[1]]))

    for nodes in hidden_layers:
        architecture.append(tf.keras.layers.Dense(nodes, activation='relu'))
        architecture.append(tf.keras.layers.Dropout(Dropout))

    # Output layer
    architecture.append(tf.keras.layers.Dense(num_classes, activation='softmax'))  # sigmoid or tanh or softplus
    # Compile the model
    model = tf.keras.Sequential(architecture)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[balanced_accuracy])
    # Get overview of model architecture
    model.summary()
    # endregion

    # region Model
    early_stopping = BalancedAccuracyEarlyStopping(
        monitor='val_balanced_accuracy',  # Monitor the balanced accuracy on the validation set
        patience=20,  # Number of epochs with no improvement
        verbose=1,
        restore_best_weights=True  # Verbosity level
    )

    history = model.fit(X_train, y_train, epochs=200, validation_data=(X_val, y_val),
                        batch_size=batch_size,
                        callbacks=[early_stopping], verbose=1, class_weight=class_weights)

    if num_classes == 5:
       save_path = os.path.join(output_path, "best_oversampling_5.png")
    elif num_classes == 2:
        save_path = os.path.join(output_path, "best_oversampling_2.png")

    plot_training_history(history, save_path=save_path, show_best_epoch=True, title="Balanced Accuracy")

    if num_classes == 5:
        model_file = os.path.join(output_path, 'best_oversampling_5_classes.keras')
    elif num_classes == 2:
        model_file = os.path.join(output_path, 'best_oversampling_2_classes.keras')

    model.save(model_file)

    return features_to_predict, model


def prediction(model, features_to_predict, data_to_predict, num_classes, output_path):
    prediction = model.predict(features_to_predict)
    numeric_predicted_labels = np.argmax(prediction, axis=1)

    species_names = data_to_predict['species'].values  # Using 'species' column for species names

    if num_classes == 5:
        label_mapping = {0 : 'LC', 1 : 'NT', 2 : 'VU', 3 : 'EN', 4 : 'CR'}
        predicted_labels = [label_mapping[idx] for idx in numeric_predicted_labels]
        mapping_2 = {'LC': 'NT', 'NT': 'NT', 'VU': 'PT', 'EN': 'PT', 'CR': 'PT'}
        predicted_labels_2 = [mapping_2[label] for label in predicted_labels]

        # Create a dataframe with species names and their corresponding predictions
        prediction_results = pd.DataFrame({
            'species': species_names,
            'RL_Class_5': predicted_labels,
            'RL_Class_2': predicted_labels_2
        })

    elif num_classes == 2:
        label_mapping = {0 : 'NT', 1 : 'PT'}
        predicted_labels = [label_mapping[idx] for idx in numeric_predicted_labels]

        # Create a dataframe with species names and their corresponding predictions
        prediction_results = pd.DataFrame({
            'species': species_names,
            'RL_Class_2': predicted_labels
        })

    assert len(predicted_labels) == len(species_names), "Mismatch between predicted labels and species names count"

    output_path = os.path.join(output_path, "predictions_best_oversampling.csv")
    prediction_results.to_csv(output_path, index=False)

    unique_values, counts = np.unique(predicted_labels, return_counts=True)
    # Display the results
    for value, count in zip(unique_values, counts):
        print(f"{value}: {count}")

    return prediction_results


def plot_bar(prediction_results, num_classes, output_path):
    # Define the class order
    if num_classes == 5:
        class_order = ['LC', 'NT', 'VU', 'EN', 'CR']

        # Count occurrences of each class
        counts = prediction_results['RL_Class_5'].value_counts()

        colors = {
            'LC': '#4CAF50',  # Green
            'NT': '#8BC34A',  # Light Green
            'VU': '#FFC107',  # Amber
            'EN': '#FF9800',  # Orange
            'CR': '#F44336'  # Red
        }

    elif num_classes == 2:
        class_order = ['NT', 'PT']

        # Count occurrences of each class
        counts = prediction_results['RL_Class_2'].value_counts()

        # Define colors for each class
        colors = {
            'NT': '#4CAF50',  # Green
            'PT': '#F44336'  # Red
        }

    # Reindex to ensure the order of classes
    counts = counts.reindex(class_order, fill_value=0)


    # Assign colors based on the class_order
    plot_colors = [colors[class_name] for class_name in class_order]

    # Plotting the bar plot
    plt.figure(figsize=(12, 7))
    bars = plt.bar(counts.index, counts.values, color=plot_colors)

    # Adding labels and title
    plt.ylabel('Number of Species', fontsize=16)

    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=12, rotation=0)
    # Add the counts on top of the bars
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(int(bar.get_height())),
                 ha='center', va='bottom', fontsize=12)

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    output_path = os.path.join(output_path, "best_oversampled_bar_plot_colored.png")
    plt.savefig(output_path)

    # Show the plot
    plt.show()

#endregion

def main (input_data, data_to_predict, hidden_layers, Dropout, batch_size,num_classes, oversampling, weight, OUTPUT_PATH):
    features_to_predict, model = training_model(input_data, data_to_predict, OUTPUT_PATH, batch_size=batch_size, Dropout=Dropout, hidden_layers=hidden_layers, num_classes=num_classes, oversampling=oversampling, weight=weight)
    predictions = prediction(model, features_to_predict, data_to_predict, num_classes, OUTPUT_PATH)
    plot_bar(predictions, num_classes, OUTPUT_PATH)


if __name__ == '__main__':
    INPUT_PATH = 'data/all_data.csv'
    input_data = pd.read_csv(INPUT_PATH)

    # Making sure that DD are excluded
    input_data['RedListCategory'] = input_data['RedListCategory'].replace('DD', np.nan)
    # Scaling all features between 0 and 1 (min and max)
    input_data = scale_variables_min_max(input_data)

    # Separating species with and without RL category
    data_to_predict = input_data[input_data['RedListCategory'].isna()]
    input_data = input_data.dropna()

    # Shuffle the data
    input_data = input_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Model architecture for the best model selected based on the balanced accuracy in the validation set
    hidden_layers = [60, 30]
    Dropout = 0.1
    batch_size = 40
    num_classes = 2
    oversampling = 'yes'  # 'yes' or 'no'
    weight = 'no'  # 'yes' or 'no'

    OUTPUT_PATH = f'results/predictions/{num_classes}_classes/'
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    main(input_data, data_to_predict, hidden_layers, Dropout, batch_size,num_classes,oversampling,weight, OUTPUT_PATH)


