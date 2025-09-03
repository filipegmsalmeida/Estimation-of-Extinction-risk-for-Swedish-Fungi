#region Importing modules
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error
import itertools
from imblearn.over_sampling import SMOTE
from keras.utils import to_categorical
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score
from tensorflow.keras.utils import register_keras_serializable

#endregion

#region Funtions
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

class EpochCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epochs = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epochs = epoch + 1  # since epoch is zero-indexed

def plot_training_history(history, title, accuracy, save_path, show_best_epoch=True, show=True):
    fig = plt.figure(figsize=(8, 5))

    if accuracy == 'balanced':
        plt.plot(history.history['balanced_accuracy'], label='Training set Balanced Acc')
        plt.plot(history.history['val_balanced_accuracy'], label='Validation set Balanced Acc')
        plt.ylabel('Balanced_acc')

        if show_best_epoch:
            best_epoch = np.argmax(history.history['val_balanced_accuracy'])
            plt.axvline(best_epoch, c='grey', linestyle='--')
            plt.axhline(history.history['val_balanced_accuracy'][best_epoch], c='grey', linestyle='--')
            plt.gca().axvspan(best_epoch, len(history.history['balanced_accuracy']), color='grey', alpha=0.3, zorder=3)

    elif accuracy =='normal':
        plt.plot(history.history['accuracy'], label='Training set Normal Acc')
        plt.plot(history.history['val_accuracy'], label='Validation set Normal Acc')
        plt.ylabel('Accuracy')

        if show_best_epoch:
            best_epoch = np.argmax(history.history['val_accuracy'])
            plt.axvline(best_epoch, c='grey', linestyle='--')
            plt.axhline(history.history['val_accuracy'][best_epoch], c='grey', linestyle='--')
            plt.gca().axvspan(best_epoch, len(history.history['accuracy']), color='grey', alpha=0.3,zorder=3)

    plt.xlabel('Epochs')
    plt.title('\n'.join(title.split()), fontsize=12)
    plt.grid()
    plt.legend(loc='upper center')

    if show == True:
        plt.show()

    fig.savefig(save_path)
    plt.close(fig)

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

def create_model(input_shape, num_classes, accuracy, layer_nodes, dropout_rate=0.0):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    for nodes in layer_nodes:
        model.add(tf.keras.layers.Dense(nodes, activation='relu'))
        if dropout_rate > 0.0:
            model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    if accuracy == "balanced":
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[balanced_accuracy])
    elif accuracy == 'normal':
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    return model

def train_model_with_cross_validation(X_combined, y_combined, settings, species):
    all_model_results = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #This makes the folds be stratified. It will keep the proportion of the classes equal in all folds

    for config in settings:
        model_results = []
        plot_paths = []

        y_combined_copy = y_combined.copy()

        if config['num_classes'] == 2:
            mapping = {'LC': 'NEN', 'NT': 'NEN', 'VU': 'EN', 'EN': 'EN', 'CR': 'EN'}
            # Translate the array using the mapping
            y_combined_copy = np.array([mapping[item] for item in y_combined])

        # Create a unique directory for each model setting
        if config['oversampling'] == 'yes' and config['weight'] == 'no':
            model_config_name = f"Oversampled_{config['num_classes']}_{config['accuracy']}_{'_'.join(map(str, config['n_layers'].split('_')))}_{config['dropout_rate']}_{config['batch_size']}"
        elif config['weight'] == 'yes' and config['oversampling'] == 'no':
            model_config_name = f"Weighted_{config['num_classes']}_{config['accuracy']}_{'_'.join(map(str, config['n_layers'].split('_')))}_{config['dropout_rate']}_{config['batch_size']}"
        elif config['oversampling'] == 'no' and config['weight'] == 'no':
            model_config_name = f"Baseline_{config['num_classes']}_{config['accuracy']}_{'_'.join(map(str, config['n_layers'].split('_')))}_{config['dropout_rate']}_{config['batch_size']}"

        model_plot_dir = os.path.join("results/train_plots", model_config_name)
        os.makedirs(model_plot_dir, exist_ok=True)

        for i, (train_index, test_index) in enumerate(skf.split(X_combined, y_combined_copy), start=1):
            X_train_val, X_test = X_combined[train_index], X_combined[test_index]
            y_train_val, y_test = y_combined_copy[train_index], y_combined_copy[test_index]
            species_train_val, species_test = species[train_index], species[test_index]

            # Further split train_val into train and validation sets
            X_train, X_val, y_train, y_val, species_train, species_val = train_test_split(X_train_val, y_train_val, species_train_val,  test_size=0.2, random_state=42, stratify=y_train_val)

            if config['oversampling'] == 'yes':
                smote = SMOTE(k_neighbors=5, random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)

            if config['num_classes'] == 5:
                mapping = {'LC': 0, 'NT': 1, 'VU': 2, 'EN': 3, 'CR': 4}
                # Translate the array using the mapping
                y_train = np.array([mapping[item] for item in y_train])
                y_train = tf.keras.utils.to_categorical(y_train)
                y_val = np.array([mapping[item] for item in y_val])
                y_val = tf.keras.utils.to_categorical(y_val)
                y_test = np.array([mapping[item] for item in y_test])
                y_test = tf.keras.utils.to_categorical(y_test)

            if config['num_classes'] == 2:
                mapping = {'NEN': 0, 'EN': 1}
                # Translate the array using the mapping
                y_train = np.array([mapping[item] for item in y_train])
                y_train = tf.keras.utils.to_categorical(y_train)
                y_val = np.array([mapping[item] for item in y_val])
                y_val = tf.keras.utils.to_categorical(y_val)
                y_test = np.array([mapping[item] for item in y_test])
                y_test = tf.keras.utils.to_categorical(y_test)

            models_test_dir = os.path.join("results/models_and_test_data")
            os.makedirs(models_test_dir, exist_ok=True)

            X_test_name = f'results/models_and_test_data/x_test_{model_config_name}_fold{i}.csv'
            y_test_name = f'results/models_and_test_data/y_test_{model_config_name}_fold{i}.csv'

            np.savetxt(X_test_name, X_test, delimiter=",")
            np.savetxt(y_test_name, y_test, delimiter=",")


            layer_nodes = [int(x) for x in config['n_layers'].split('_')]
            dropout_rate = config['dropout_rate']
            batch_size = config['batch_size']
            accuracy = config['accuracy']

            model_instance = create_model(input_shape=(X_combined.shape[1],), num_classes=config['num_classes'],
                                          layer_nodes=layer_nodes, dropout_rate=dropout_rate,
                                          accuracy= accuracy)

            print(f"Model created for fold {i}")

            model_instance.summary()

            class_weights = None
            if config['weight'] == 'yes' and config['oversampling'] == 'no':
                class_weights = class_weight.compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(np.argmax(y_train, axis=1)),
                    y=np.argmax(y_train, axis=1)
                )
                class_weights = dict(enumerate(class_weights))
                print(f"Class weights for fold {i}: {class_weights}")



            if accuracy == 'balanced':
                early_stopping = BalancedAccuracyEarlyStopping(
                    monitor='val_balanced_accuracy',
                    # Monitor the balanced accuracy on the validation set
                    patience=20,  # Number of epochs with no improvement
                    verbose=1,
                    restore_best_weights=True  # Verbosity level
                )
            elif accuracy == 'normal':
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=20,
                    restore_best_weights=True,
                )

            # Train the model
            history = model_instance.fit(X_train, y_train, epochs=200, validation_data=(X_val, y_val),
                                         batch_size=batch_size, class_weight=class_weights,
                                         callbacks=[early_stopping], verbose=1)

            model_file = f'results/models_and_test_data/model_{model_config_name}_fold{i}.keras'
            model_instance.save(model_file)

            plot_title = f"Fold: {i}/{5}"
            plot_path = os.path.join(model_plot_dir, f'plot_fold_{i}.png')
            plot_training_history(history, show_best_epoch=True, accuracy=accuracy, title=plot_title, save_path = plot_path, show=False)
            plot_paths.append(plot_path)

            if accuracy == 'normal':
                best_epoch = np.argmax(history.history['val_accuracy'])
                best_val_metric = history.history['val_accuracy'][best_epoch]
            elif accuracy == 'balanced':
                best_epoch = np.argmax(history.history['val_balanced_accuracy'])
                best_val_metric = history.history['val_balanced_accuracy'][best_epoch]


            test_loss, test_metric = model_instance.evaluate(X_test, y_test, verbose=1)

            y_pred = model_instance.predict(X_test)


            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)

            pred_species = pd.DataFrame({
                "species": species_test,
                "y_true": y_true_classes,
                "y_pred": y_pred_classes
            })
            pred_species.to_csv(f'results/models_and_test_data/predictions_{model_config_name}_fold{i}.csv', index=False)

            test_accuracy = accuracy_score(y_true_classes, y_pred_classes)
            print(f'Test accuracy{test_metric}')

            balanced_acc = balanced_accuracy_score(y_true_classes, y_pred_classes)
            print(f'balanced_accuracy{balanced_acc}')

            # Calculate per-class accuracy
            class_accuracies = {}
            class_labels = np.unique(y_true_classes)
            for label in class_labels:
                class_indices = np.where(y_true_classes == label)[0]
                class_accuracies[label] = accuracy_score(y_true_classes[class_indices],
                                                             y_pred_classes[class_indices])

            # Calculate class weights
            class_weights_test = {}
            total_samples = len(y_true_classes)
            for label in class_labels:
                class_weights_test[label] = np.sum(y_true_classes == label) / total_samples

            print(f'class_frequency{class_weights_test}')

            classweights_array = np.array(list(class_weights_test.values()))
            inverted_array = 1 / classweights_array
            scaled_inverted_weights_array = inverted_array / max(inverted_array)

            print(f'scalad_weights{scaled_inverted_weights_array}')

            # Compute weighted accuracy
            weighted_accuracy = sum(class_accuracies[label] * scaled_inverted_weights_array[label] for label in class_labels)
            print(f"Weighted accuracy for fold {i}: {weighted_accuracy}")

            # Compute confusion matrix
            cm = confusion_matrix(y_true_classes, y_pred_classes)

            mcc_test = matthews_corrcoef(y_true_classes, y_pred_classes)

            # Compute F1 score
            f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')


            model_results.append({
                'fold': i,
                'best_epoch': best_epoch,
                'best_val_metric': best_val_metric,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'confusion_matrix': cm,
                'f1_score': f1,
                'plot_path': plot_path,
                'mcc_test': mcc_test,
                'balanced_acc': balanced_acc
            })

        # Calculate mean results for each model configuration
        def safe_mean(values):
            valid_values = [v for v in values if v is not None]
            return np.mean(valid_values) if valid_values else float('nan')

        mean_best_epoch = np.mean([result['best_epoch'] for result in model_results])
        mean_val_metric = safe_mean([result['best_val_metric'] for result in model_results])
        mean_test_accuracy = np.mean([result['test_accuracy'] for result in model_results])
        mean_test_loss = np.mean([result['test_loss'] for result in model_results])
        mean_f1_score = np.mean([result['f1_score'] for result in model_results])
        mean_mcc_test = np.mean([result['mcc_test'] for result in model_results])
        mean_balanced_acc = np.mean([result['balanced_acc'] for result in model_results])

        all_model_results.append({
            'config': config,
            'fold_results': model_results,
            'mean_results': {
                'mean_best_epoch': mean_best_epoch,
                'mean_val_metric': mean_val_metric,
                'mean_test_accuracy': mean_test_accuracy,
                'mean_test_loss': mean_test_loss,
                'mean_f1_score': mean_f1_score,
                'mean_mcc_test': mean_mcc_test,
                'mean_balanced_acc': mean_balanced_acc
            },
            'plot_paths': plot_paths
        })

    return all_model_results

def print_mean_results(all_model_results):
    results = []
    for model_result in all_model_results:
        config = model_result['config']
        mean_results = {
            'num_classes': config['num_classes'],
            'accuracy': config['accuracy'],
            'n_layers': config['n_layers'],
            'dropout_rate': config['dropout_rate'],
            'batch_size': config['batch_size'],
            'oversampling': config['oversampling'],
            'weight': config['weight'],
            'mean_val_metric': model_result['mean_results']['mean_val_metric'],
            'mean_best_epoch': model_result['mean_results']['mean_best_epoch'],
            'mean_f1_score': model_result['mean_results']['mean_f1_score'],
            'mean_test_accuracy': model_result['mean_results']['mean_test_accuracy'],
            'mean_mcc_test': model_result['mean_results']['mean_mcc_test'],
            'mean_balanced_acc': model_result['mean_results']['mean_balanced_acc']
        }

        results.append(mean_results)

    df = pd.DataFrame(results)

    # Separate DataFrames based on  num_classes, and oversampling
    oversampled_normal_acc_5 = df[(df['num_classes'] == 5) & (df['oversampling'] == 'yes') & (df['weight'] == 'no') & (df['accuracy'] == 'normal')]
    baseline_normal_acc_5 = df[(df['num_classes'] == 5) & (df['oversampling'] == 'no') & (df['weight'] == 'no') & (df['accuracy'] == 'normal')]
    baseline_balanced_acc_5 = df[(df['num_classes'] == 5) & (df['oversampling'] == 'no') & (df['weight'] == 'no') & (df['accuracy'] == 'balanced')]
    oversampled_balanced_acc_5 = df[(df['num_classes'] == 5) & (df['oversampling'] == 'yes') & (df['weight'] == 'no') & (df['accuracy'] == 'balanced')]
    weighted_normal_acc_5 = df[(df['num_classes'] == 5) & (df['oversampling'] == 'no') & (df['weight'] == 'yes') & (df['accuracy'] == 'normal')]
    weighted_balanced_acc_5 = df[(df['num_classes'] == 5) & (df['oversampling'] == 'no') & (df['weight'] == 'yes') & (df['accuracy'] == 'balanced')]
    baseline_normal_acc_2 = df[(df['num_classes'] == 2) & (df['oversampling'] == 'no') & (df['weight'] == 'no') & (df['accuracy'] == 'normal')]
    baseline_balanced_acc_2 = df[(df['num_classes'] == 2) & (df['oversampling'] == 'no') & (df['weight'] == 'no') & (df['accuracy'] == 'balanced')]
    oversampled_normal_acc_2 = df[(df['num_classes'] == 2) & (df['oversampling'] == 'yes') & (df['weight'] == 'no') & (df['accuracy'] == 'normal')]
    oversampled_balanced_acc_2 = df[(df['num_classes'] == 2) & (df['oversampling'] == 'yes') & (df['weight'] == 'no') & (df['accuracy'] == 'balanced')]
    weighted_normal_acc_2 = df[(df['num_classes'] == 2) & (df['oversampling'] == 'no') & (df['weight'] == 'yes') & (df['accuracy'] == 'normal')]
    weighted_balanced_acc_2 = df[(df['num_classes'] == 2) & (df['oversampling'] == 'no') & (df['weight'] == 'yes') & (df['accuracy'] == 'balanced')]

    return weighted_balanced_acc_2, weighted_normal_acc_2, oversampled_balanced_acc_2, oversampled_normal_acc_2, oversampled_normal_acc_5, oversampled_balanced_acc_5, weighted_normal_acc_5, weighted_balanced_acc_5, oversampled_normal_acc_5, baseline_normal_acc_5, baseline_balanced_acc_5, baseline_normal_acc_2, baseline_balanced_acc_2

def visualize_confusion_matrix_highest_test_accuracy(all_model_results, labels, num_classes, oversampling, weight, accuracy, save_path, save=False, show=True):
    # Filter all_model_results based on  num_classes, and oversampling
    model_results_filtered = [result for result in all_model_results if result['config']['num_classes'] == num_classes and result['config']['oversampling'] == oversampling and result['config']['weight'] == weight and result['config']['accuracy'] == accuracy]

    if not model_results_filtered:
        print(f"No models with {num_classes} classes and oversampling = {oversampling} found.")
        return

    # Find the model configuration with the highest mean test accuracy for classification or lowest mean_val_metric for regression

    best_model_result = max(model_results_filtered, key=lambda x: x['mean_results']['mean_val_metric'])

    # Extract model settings of the best model
    best_model_settings = best_model_result['config']
    best_model_name = f"{best_model_settings['n_layers']}_{best_model_settings['dropout_rate']}_{best_model_settings['batch_size']}_{best_model_settings['oversampling']}_{best_model_settings['weight']}_{best_model_settings['accuracy']}"

    # Extract confusion matrices for the best model
    confusion_matrices = [fold_result['confusion_matrix'] for fold_result in best_model_result['fold_results']]

    # Find the maximum shape among all confusion matrices
    max_shape = max(cm.shape for cm in confusion_matrices)

    # Pad or resize each confusion matrix to have the same shape as the maximum shape
    padded_confusion_matrices = [
        np.pad(cm, [(0, max_shape[0] - cm.shape[0]), (0, max_shape[1] - cm.shape[1])], mode='constant') for cm in
        confusion_matrices]

    # Combine all confusion matrices across folds
    combined_confusion_matrix = np.sum(padded_confusion_matrices, axis=0)

    # Calculate row-wise sum (total samples for each true label)
    row_sums = combined_confusion_matrix.sum(axis=1, keepdims=True)

    # Normalize the combined confusion matrix
    combined_confusion_matrix_percent = combined_confusion_matrix / row_sums

    # Plot the combined confusion matrix with percentages
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.set(font_scale=1.2)  # Adjust font size for better readability
    sns.heatmap(combined_confusion_matrix_percent, annot=combined_confusion_matrix, fmt="d", cmap="Blues", cbar=True,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 12, 'weight': 'bold'},
                linewidths=1, linecolor='gray', ax=ax)
    ax.xaxis.set_label_position('top')  # Move xlabel to the top
    ax.set_xlabel('Predicted Label', weight='bold')
    ax.set_ylabel('True Label', weight='bold')
    ax.xaxis.tick_top()  # Move x-axis labels to the top
    ax.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
    ax.tick_params(axis='y', left=True)
    plt.figtext(0.5, 0.04, f"{best_model_name}", ha='center', fontsize=14, va='center')

    if show:
        plt.show()

    if save:
        fig.savefig(save_path)
        plt.close(fig)

#endregion

def main (features,labels, settings, run_name, species):
    #region Test models using Cross Validation
    all_models_result = train_model_with_cross_validation(features, labels, settings, species)

    #SAVE IT
    dir_path_results_cv = "results/save_model/"
    os.makedirs(dir_path_results_cv, exist_ok=True)
    file_path_results = os.path.join(dir_path_results_cv, f"{run_name}.pkl") #Always change it
    with open(file_path_results, "wb") as pickle_file:
        pickle.dump(all_models_result, pickle_file)

    #Summaring models
    weighted_balanced_acc_2, weighted_normal_acc_2, oversampled_balanced_acc_2, oversampled_normal_acc_2, oversampled_normal_acc_5, oversampled_balanced_acc_5, weighted_normal_acc_5, weighted_balanced_acc_5, oversampled_normal_acc_5, baseline_normal_acc_5, baseline_balanced_acc_5, baseline_normal_acc_2, baseline_balanced_acc_2 = print_mean_results(all_models_result)

    #SAVE IT
    dir_path_tables_cv = "results/tables"
    os.makedirs(dir_path_tables_cv, exist_ok=True)

    result_tables = {
        'baseline_normal_acc_5': baseline_normal_acc_5,
        'baseline_balanced_acc_5': baseline_balanced_acc_5,
        'oversampled_normal_acc_5': oversampled_normal_acc_5,
        'oversampled_balanced_acc_5': oversampled_balanced_acc_5,
        'weighted_normal_acc_5': weighted_normal_acc_5,
        'weighted_balanced_acc_5': weighted_balanced_acc_5,
        'oversampled_normal_acc_2': oversampled_normal_acc_2,
        'oversampled_balanced_acc_2': oversampled_balanced_acc_2,
        'weighted_normal_acc_2': weighted_normal_acc_2,
        'weighted_balanced_acc_2': weighted_balanced_acc_2,
        'baseline_normal_acc_2': baseline_normal_acc_2,
        'baseline_balanced_acc_2': baseline_balanced_acc_2
    }

    for name, table in result_tables.items():
        if not table.empty:
            table.to_csv(os.path.join(dir_path_tables_cv, f'{name}.csv'), index=False)
        else:
            print(f'Skipping saving {name}.csv as it is empty')

    # Confusion matrix
    dir_path_cm_cv = "results/confusion_matrix"
    os.makedirs(dir_path_cm_cv, exist_ok=True)

    labels_classification_5 = ['LC', 'NT', 'VU', 'EN', 'CR']
    labels_classification_2 = ['Not Threatened', 'Possibly Threatened']

    #Best test accuracy
    for num_classes in [5, 2]:
        for oversampled in ['yes', 'no']:
            for weight in ['yes', 'no']:
                for accuracy in ['balanced', 'normal']:

                    labels = labels_classification_5 if num_classes == 5 else labels_classification_2

                    # Determine sampling status
                    if oversampled == "yes" and weight == "no":
                        sampling_status = "oversampled"
                    elif weight == "no" and oversampled == "no":
                        sampling_status = "baseline"
                    elif weight == "yes" and oversampled == "no":
                        sampling_status = "weighted"
                    else:
                        print(f"Skipping combination: oversampled={oversampled}, weight={weight}")
                        continue

                    filename = f"{sampling_status}_{num_classes}_classes_{accuracy}_confusion_matrix_best_val_metric.png"
                    save_path = os.path.join(dir_path_cm_cv, filename)

                    visualize_confusion_matrix_highest_test_accuracy(
                        all_models_result, labels, num_classes, oversampled, weight, accuracy,
                        save_path=save_path, save=True, show=False
                    )

    #endregion

if __name__ == '__main__':
    # Inputing data
    input_data_path = 'data/all_data.csv'
    input_data = pd.read_csv(input_data_path)

    # Making sure that DD are excluded
    input_data['RedListCategory'] = input_data['RedListCategory'].replace('DD', np.nan)

    # Scaling all features between 0 and 1 (min and max)
    input_data = scale_variables_min_max(input_data)

    # Separating species with and without RL category
    RL_na = input_data[input_data['RedListCategory'].isna()]
    input_data = input_data.dropna()

    # Shuffle the data
    input_data = input_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Defining features and labels
    features = input_data.values[:, 1:-1].astype(float)
    feature_names = input_data.columns[1:-1]
    labels = input_data.values[:, -1]
    species = input_data.iloc[:, 0].values  # species names

    unique_values, counts = np.unique(labels, return_counts=True)
    # Display the results
    for value, count in zip(unique_values, counts):
        print(f"{value}: {count}")
    # endregion

    # Settings
    dropout_rates = [0.1, 0.2, 0.3]
    n_layers = ['60', '60_30', '60_30_15', '30', '30_15', '30_15_8']
    batch_size = [10, 20, 40]
    oversampling = ['no', 'yes']  # ['yes', 'no'] If yes it will be oversampled, if no it will be weighted or baseline
    weight = ['no', 'yes']  # ['yes', 'no'] If no it will be baseline, if yes it will be weighted
    num_classes = [2]  # 2 or 5  CHANGE NAME TO SAVE THE FINAL RESULTS
    accuracy = ['balanced']  # balanced or normal CHANGE NAME TO SAVE THE FINAL RESULTS
    settings = [
        {'n_layers': n, 'dropout_rate': d, 'batch_size': b, 'oversampling': o, 'num_classes': nc, 'accuracy': acc,
         'weight': w} for n, d, b, o, nc, acc, w in
        itertools.product(n_layers, dropout_rates, batch_size, oversampling, num_classes, accuracy, weight)]

    # Name to save the type of run
    run_name = f"{accuracy}_accuracy_{num_classes}"

    # NEED TO CHANGE THE SMOTE PARAMETERS FOR THE FINAL RUN

    main(features,labels, settings, run_name, species)

