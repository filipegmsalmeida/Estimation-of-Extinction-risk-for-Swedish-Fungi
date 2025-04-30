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

def plot_training_history(history, title, accuracy, save_path, model_type='classification', show_best_epoch=True, show=True):
    fig = plt.figure(figsize=(8, 5))

    if model_type == 'classification':
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

    elif model_type == 'regression':
        plt.plot(history.history['mae'], label='Training set MAE')
        plt.plot(history.history['val_mae'], label='Validation set MAE')
        plt.ylabel('Mean Absolute Error (MAE)')

        if show_best_epoch:
            best_epoch = np.argmin(history.history['val_mae'])
            plt.axvline(best_epoch, c='grey', linestyle='--')
            plt.axhline(history.history['val_mae'][best_epoch], c='grey', linestyle='--')
            plt.gca().axvspan(best_epoch, len(history.history['mae']), color='grey', alpha=0.3, zorder=3)

    plt.xlabel('Epochs')
    plt.title('\n'.join(title.split()), fontsize=12)
    plt.grid()
    plt.legend(loc='upper center')

    if show == True:
        plt.show()

    fig.savefig(save_path)
    plt.close(fig)

def map_to_status(prediction, num_classes):
    if num_classes == 5:
        if prediction <= 0.2: return 0
        elif prediction <= 0.4: return 1
        elif prediction <= 0.6: return 2
        elif prediction <= 0.8: return 3
        else: return 4
    elif num_classes == 2:
        if prediction <= 0.5:return 0
        else: return 1

    else:
        raise ValueError("Unsupported number of classes")

def custom_accuracy(y_true, y_pred, num_classes):
    y_pred_classes = [map_to_status(pred, num_classes) for pred in y_pred]
    y_true_classes = [map_to_status(pred, num_classes) for pred in y_true]
    correct_predictions = sum(1 for true, pred in zip(y_true_classes, y_pred_classes) if true == pred)
    return correct_predictions / len(y_true)

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

def create_model(input_shape, num_classes, accuracy, layer_nodes, dropout_rate=0.0, model_type = 'classification'):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    for nodes in layer_nodes:
        model.add(tf.keras.layers.Dense(nodes, activation='relu'))
        if dropout_rate > 0.0:
            model.add(tf.keras.layers.Dropout(dropout_rate))

    if model_type == 'classification':
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
        if accuracy == "balanced":
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[balanced_accuracy])
        elif accuracy == 'normal':
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    elif model_type == 'regression':
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='mae', optimizer='adam', metrics=['mae'])

    return model

def train_model_with_cross_validation(X_combined, y_combined, settings):
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
        if config['oversampling'] == 'yes':
            model_config_name = f"Oversampled_{config['model_type']}_{config['num_classes']}_{config['accuracy']}_{'_'.join(map(str, config['n_layers'].split('_')))}_{config['dropout_rate']}_{config['batch_size']}"
        elif config['oversampling'] == 'no':
            model_config_name = f"{config['model_type']}_{config['num_classes']}_{config['accuracy']}_{'_'.join(map(str, config['n_layers'].split('_')))}_{config['dropout_rate']}_{config['batch_size']}"

        model_plot_dir = os.path.join("results/final_models_report/train_plots", model_config_name)
        os.makedirs(model_plot_dir, exist_ok=True)

        for i, (train_index, test_index) in enumerate(skf.split(X_combined, y_combined_copy), start=1):
            X_train_val, X_test = X_combined[train_index], X_combined[test_index]
            y_train_val, y_test = y_combined_copy[train_index], y_combined_copy[test_index]

            # Further split train_val into train and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)

            if config['oversampling'] == 'yes':
                smote = SMOTE(k_neighbors=5, random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)

            if config['model_type'] == 'classification' and config['num_classes'] == 5:
                mapping = {'LC': 0, 'NT': 1, 'VU': 2, 'EN': 3, 'CR': 4}
                # Translate the array using the mapping
                y_train = np.array([mapping[item] for item in y_train])
                y_train = tf.keras.utils.to_categorical(y_train)
                y_val = np.array([mapping[item] for item in y_val])
                y_val = tf.keras.utils.to_categorical(y_val)
                y_test = np.array([mapping[item] for item in y_test])
                y_test = tf.keras.utils.to_categorical(y_test)

            if config['model_type'] == 'regression' and config['num_classes'] == 5:
                mapping = {'LC': 0.1, 'NT': 0.3, 'VU': 0.5, 'EN': 0.7, 'CR': 0.9}
                # Translate the array using the mapping
                y_train = np.array([mapping[item] for item in y_train])
                y_val = np.array([mapping[item] for item in y_val])
                y_test = np.array([mapping[item] for item in y_test])

            if config['model_type'] == 'regression' and config['num_classes'] == 2:
                mapping = {'NEN': 0.25, 'EN': 0.75}
                # Translate the array using the mapping
                y_train = np.array([mapping[item] for item in y_train])
                y_val = np.array([mapping[item] for item in y_val])
                y_test = np.array([mapping[item] for item in y_test])

            if config['model_type'] == 'classification' and config['num_classes'] == 2:
                mapping = {'NEN': 0, 'EN': 1}
                # Translate the array using the mapping
                y_train = np.array([mapping[item] for item in y_train])
                y_train = tf.keras.utils.to_categorical(y_train)
                y_val = np.array([mapping[item] for item in y_val])
                y_val = tf.keras.utils.to_categorical(y_val)
                y_test = np.array([mapping[item] for item in y_test])
                y_test = tf.keras.utils.to_categorical(y_test)

            models_test_dir = os.path.join("results/final_models_report/models_and_test_data")
            os.makedirs(models_test_dir, exist_ok=True)

            X_test_name = f'results/final_models_report/models_and_test_data/x_test_{model_config_name}_fold{i}.csv'
            y_test_name = f'results/final_models_report/models_and_test_data/y_test_{model_config_name}_fold{i}.csv'

            np.savetxt(X_test_name, X_test, delimiter=",")
            np.savetxt(y_test_name, y_test, delimiter=",")


            layer_nodes = [int(x) for x in config['n_layers'].split('_')]
            dropout_rate = config['dropout_rate']
            batch_size = config['batch_size']
            model_type = config['model_type']
            accuracy = config['accuracy']

            model_instance = create_model(input_shape=(X_combined.shape[1],), num_classes=config['num_classes'],
                                          layer_nodes=layer_nodes, dropout_rate=dropout_rate, model_type=model_type,
                                          accuracy= accuracy)

            print(f"Model created for fold {i}")

            model_instance.summary()

            class_weights = None

            if config['oversampling'] == 'no' and model_type == 'classification':
                # Calculate class weights for the fold
                class_weights = class_weight.compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(np.argmax(y_train, axis=1)),
                    y=np.argmax(y_train, axis=1)
                )
                class_weights = dict(enumerate(class_weights))
                print(f"Class weights for fold {i}: {class_weights}")

            if model_type == 'classification':
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

            if model_type =='regression':
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_mae',
                    patience=20,
                    restore_best_weights=True,
                )

            # Train the model
            history = model_instance.fit(X_train, y_train, epochs=200, validation_data=(X_val, y_val),
                                         batch_size=batch_size, class_weight=class_weights,
                                         callbacks=[early_stopping], verbose=1)

            model_file = f'results/final_models_report/models_and_test_data/model_{model_config_name}_fold{i}.keras'
            model_instance.save(model_file)

            plot_title = f"Fold: {i}/{5}"
            plot_path = os.path.join(model_plot_dir, f'plot_fold_{i}.png')
            plot_training_history(history, model_type=model_type, show_best_epoch=True, accuracy=accuracy, title=plot_title, save_path = plot_path, show=False)
            plot_paths.append(plot_path)

            if accuracy == 'normal':
                best_epoch = np.argmax(history.history['val_accuracy']) if config['model_type'] == 'classification' else np.argmin(history.history['val_mae'])
                best_val_metric = history.history['val_accuracy'][best_epoch] if config['model_type'] == 'classification' else history.history['val_mae'][best_epoch]
            elif accuracy == 'balanced':
                best_epoch = np.argmax(history.history['val_balanced_accuracy']) if config['model_type'] == 'classification' else np.argmin(history.history['val_mae'])
                best_val_metric = history.history['val_balanced_accuracy'][best_epoch] if config['model_type'] == 'classification' else history.history['val_mae'][best_epoch]


            test_loss, test_metric = model_instance.evaluate(X_test, y_test, verbose=1)

            y_pred = model_instance.predict(X_test)

            if config['model_type'] == 'classification':
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true_classes = np.argmax(y_test, axis=1)
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

            else:
                test_accuracy = custom_accuracy(y_test, y_pred.flatten(), config['num_classes'])
                y_pred_classes = [map_to_status(pred, config['num_classes']) for pred in y_pred.flatten()]
                y_true_classes = [map_to_status(true, config['num_classes']) for true in y_test]

                balanced_acc = balanced_accuracy_score(y_true_classes, y_pred_classes)

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
            'model_type': config['model_type'],  # Add model_type to mean_results
            'num_classes': config['num_classes'],
            'accuracy': config['accuracy'],
            'n_layers': config['n_layers'],
            'dropout_rate': config['dropout_rate'],
            'batch_size': config['batch_size'],
            'oversampling': config['oversampling'],  # Add oversampling to mean_results
            'mean_val_metric': model_result['mean_results']['mean_val_metric'],
            'mean_best_epoch': model_result['mean_results']['mean_best_epoch'],
            'mean_f1_score': model_result['mean_results']['mean_f1_score'],
            'mean_test_accuracy': model_result['mean_results']['mean_test_accuracy'],
            'mean_mcc_test': model_result['mean_results']['mean_mcc_test'],
            'mean_balanced_acc': model_result['mean_results']['mean_balanced_acc']
        }

        results.append(mean_results)

    df = pd.DataFrame(results)

    # Separate DataFrames based on model_type, num_classes, and oversampling
    classification_5_oversampled_normal_acc = df[(df['model_type'] == 'classification') & (df['num_classes'] == 5) & (df['oversampling'] == 'yes') & (df['accuracy'] == 'normal')]
    classification_5_oversampled_balanced_acc = df[(df['model_type'] == 'classification') & (df['num_classes'] == 5) & (df['oversampling'] == 'yes') & (df['accuracy'] == 'balanced')]
    classification_5_non_oversampled_normal_acc = df[(df['model_type'] == 'classification') & (df['num_classes'] == 5) & (df['oversampling'] == 'no') & (df['accuracy'] == 'normal')]
    classification_5_non_oversampled_balanced_acc = df[(df['model_type'] == 'classification') & (df['num_classes'] == 5) & (df['oversampling'] == 'no') & (df['accuracy'] == 'balanced')]
    classification_2_oversampled_normal_acc = df[(df['model_type'] == 'classification') & (df['num_classes'] == 2) & (df['oversampling'] == 'yes') & (df['accuracy'] == 'normal')]
    classification_2_oversampled_balanced_acc = df[(df['model_type'] == 'classification') & (df['num_classes'] == 2) & (df['oversampling'] == 'yes') & (df['accuracy'] == 'balanced')]
    classification_2_non_oversampled_normal_acc = df[(df['model_type'] == 'classification') & (df['num_classes'] == 2) & (df['oversampling'] == 'no') & (df['accuracy'] == 'normal')]
    classification_2_non_oversampled_balanced_acc = df[(df['model_type'] == 'classification') & (df['num_classes'] == 2) & (df['oversampling'] == 'no') & (df['accuracy'] == 'balanced')]
    regression_5_oversampled = df[(df['model_type'] == 'regression') & (df['num_classes'] == 5) & (df['oversampling'] == 'yes')]
    regression_5_non_oversampled = df[(df['model_type'] == 'regression') & (df['num_classes'] == 5) & (df['oversampling'] == 'no')]
    regression_2_oversampled = df[(df['model_type'] == 'regression') & (df['num_classes'] == 2) & (df['oversampling'] == 'yes')]
    regression_2_non_oversampled = df[(df['model_type'] == 'regression') & (df['num_classes'] == 2) & (df['oversampling'] == 'no')]

    return classification_2_non_oversampled_balanced_acc, classification_2_non_oversampled_normal_acc, classification_2_oversampled_balanced_acc, classification_5_oversampled_normal_acc, classification_5_oversampled_balanced_acc, classification_5_non_oversampled_normal_acc, classification_5_non_oversampled_balanced_acc, classification_2_oversampled_normal_acc, regression_5_oversampled, regression_5_non_oversampled, regression_2_oversampled, regression_2_non_oversampled

def visualize_confusion_matrix_highest_test_accuracy(all_model_results, labels, model_type, num_classes, oversampling, accuracy, save_path, save=False, show=True):
    # Filter all_model_results based on model_type, num_classes, and oversampling
    model_results_filtered = [result for result in all_model_results if result['config']['model_type'] == model_type and result['config']['num_classes'] == num_classes and result['config']['oversampling'] == oversampling and result['config']['accuracy'] == accuracy]

    if not model_results_filtered:
        print(f"No {model_type} models with {num_classes} classes and oversampling = {oversampling} found.")
        return

    # Find the model configuration with the highest mean test accuracy for classification or lowest mean_val_metric for regression
    if model_type == 'classification':
        best_model_result = max(model_results_filtered, key=lambda x: x['mean_results']['mean_test_accuracy'])
    else:  # regression
        best_model_result = min(model_results_filtered, key=lambda x: x['mean_results']['mean_val_metric'])

    # Extract model settings of the best model
    best_model_settings = best_model_result['config']
    best_model_name = f"{best_model_settings['model_type']}_{best_model_settings['n_layers']}_{best_model_settings['dropout_rate']}_{best_model_settings['batch_size']}_{best_model_settings['oversampling']}"

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

def visualize_confusion_matrix_highest_F1_score(all_model_results, labels, model_type, num_classes, oversampling, accuracy, save_path, save=False, show=True):
    # Filter all_model_results based on model_type, num_classes, and oversampling
    model_results_filtered = [result for result in all_model_results if result['config']['model_type'] == model_type and result['config']['num_classes'] == num_classes and result['config']['oversampling'] == oversampling and result['config']['accuracy'] == accuracy]

    if not model_results_filtered:
        print(f"No {model_type} models with {num_classes} classes and oversampling = {oversampling} found.")
        return

    # Find the model configuration with the highest mean test accuracy for classification or lowest mean_val_metric for regression
    if model_type == 'classification':
        best_model_result = max(model_results_filtered, key=lambda x: x['mean_results']['mean_balanced_acc'])
    else:  # regression
        best_model_result = min(model_results_filtered, key=lambda x: x['mean_results']['mean_balanced_acc'])

    # Extract model settings of the best model
    best_model_settings = best_model_result['config']
    best_model_name = f"{best_model_settings['model_type']}_{best_model_settings['n_layers']}_{best_model_settings['dropout_rate']}_{best_model_settings['batch_size']}_{best_model_settings['oversampling']}"

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

def visualize_confusion_matrix_highest_MCC_score(all_model_results, labels, model_type, num_classes, oversampling, accuracy, save_path, save=False, show=True):
    # Filter all_model_results based on model_type, num_classes, and oversampling
    model_results_filtered = [result for result in all_model_results if result['config']['model_type'] == model_type and result['config']['num_classes'] == num_classes and result['config']['oversampling'] == oversampling and result['config']['accuracy'] == accuracy]

    if not model_results_filtered:
        print(f"No {model_type} models with {num_classes} classes and oversampling = {oversampling} found.")
        return

    # Find the model configuration with the highest mean test accuracy for classification or lowest mean_val_metric for regression
    if model_type == 'classification':
        best_model_result = max(model_results_filtered, key=lambda x: x['mean_results']['mean_mcc_test'])
    else:  # regression
        best_model_result = min(model_results_filtered, key=lambda x: x['mean_results']['mean_mcc_test'])

    # Extract model settings of the best model
    best_model_settings = best_model_result['config']
    best_model_name = f"{best_model_settings['model_type']}_{best_model_settings['n_layers']}_{best_model_settings['dropout_rate']}_{best_model_settings['batch_size']}_{best_model_settings['oversampling']}"

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

#region Input and preparing data
#Inputing data
input_data_path = '/data/subset_all_data.csv'
input_data = pd.read_csv(input_data_path)

#Excluding ground cover data. This is because it is a categorical variable. Here I have the mean for each raster per occurrence per species. This is not right.
input_data = input_data.drop(columns=['groundcover_max', 'groundcover_min'])

#Making sure that DD are excluded
input_data['RedListCategory'] = input_data['RedListCategory'].replace('DD', np.nan)

#Scaling all features between 0 and 1 (min and max)
input_data = scale_variables_min_max(input_data)

#Scaling all features between 0 and 1 (mean)
#input_data = scale_variables_mean(input_data)

#Separating species with and without RL category
RL_na = input_data[input_data.iloc[:, 36:].isna().any(axis=1)]
input_data = input_data.dropna()

# Shuffle the data
input_data = input_data.sample(frac=1, random_state=42).reset_index(drop=True)

#Defining features and labels
features = input_data.values[:,1:-1].astype(float)
feature_names = input_data.columns[1:-1]
labels = input_data.values[:,-1]

unique_values, counts = np.unique(labels, return_counts=True)
# Display the results
for value, count in zip(unique_values, counts):
    print(f"{value}: {count}")
#endregion

#Settings
dropout_rates = [0.1, 0.2, 0.3]
n_layers = ['60', '60_30', '60_30_15', '30', '30_15', '30_15_8']
batch_size = [10, 20, 40]
model_type = ['classification'] # classification or regression CHANGE NAME TO SAVE THE FINAL RESULTS
oversampling = ['yes', 'no']
num_classes = [5] # 2 or 5  CHANGE NAME TO SAVE THE FINAL RESULTS
accuracy = ['balanced'] # balanced or normal CHANGE NAME TO SAVE THE FINAL RESULTS
settings = [{'n_layers': n, 'dropout_rate': d, 'batch_size': b, 'model_type': m, 'oversampling': o, 'num_classes': nc, 'accuracy': acc} for n, d, b, m, o, nc, acc in itertools.product(n_layers, dropout_rates, batch_size, model_type, oversampling, num_classes, accuracy)]

run_with_cross_validation = True

if run_with_cross_validation:
    #region Test models using Cross Validation
    all_models_result = train_model_with_cross_validation(features, labels, settings)

    #SAVE IT
    dir_path_results_cv = "results/final_models_report"
    os.makedirs(dir_path_results_cv, exist_ok=True)
    file_path_results = os.path.join(dir_path_results_cv, f"results_classification_5_balanced.pkl") #Always change it
    with open(file_path_results, "wb") as pickle_file:
        pickle.dump(all_models_result, pickle_file)

    # To open it
    #file_path = "C:/Users/filip/Desktop/Analysis_Fungi_Conservation/dowload_variables/NN_model/models/results/final_models_report/results_classification_2_balanced.pkl"
    #with open(file_path, "rb") as pickle_file:
    #   all_models_result = pickle.load(pickle_file)

    #Summaring models
    classification_2_non_oversampled_balanced_acc, classification_2_non_oversampled_normal_acc, classification_2_oversampled_balanced_acc, classification_5_oversampled_normal_acc, classification_5_oversampled_balanced_acc, classification_5_non_oversampled_normal_acc, classification_5_non_oversampled_balanced_acc, classification_2_oversampled_normal_acc, regression_5_oversampled, regression_5_non_oversampled, regression_2_oversampled, regression_2_non_oversampled = print_mean_results(all_models_result)

    #SAVE IT
    dir_path_tables_cv = "results/final_models_report/tables"
    os.makedirs(dir_path_tables_cv, exist_ok=True)

    result_tables = {
        'classification_5_oversampled_normal_acc': classification_5_oversampled_normal_acc,
        'classification_5_oversampled_balanced_acc': classification_5_oversampled_balanced_acc,
        'classification_5_non_oversampled_normal_acc': classification_5_non_oversampled_normal_acc,
        'classification_5_non_oversampled_balanced_acc': classification_5_non_oversampled_balanced_acc,
        'classification_2_oversampled_normal_acc': classification_2_oversampled_normal_acc,
        'classification_2_oversampled_balanced_acc': classification_2_oversampled_balanced_acc,
        'classification_2_non_oversampled_normal_acc': classification_2_non_oversampled_normal_acc,
        'classification_2_non_oversampled_balanced_acc': classification_2_non_oversampled_balanced_acc,
        'regression_5_oversampled': regression_5_oversampled,
        'regression_5_non_oversampled': regression_5_non_oversampled,
        'regression_2_oversampled': regression_2_oversampled,
        'regression_2_non_oversampled': regression_2_non_oversampled
    }

    for name, table in result_tables.items():
        if not table.empty:
            table.to_csv(os.path.join(dir_path_tables_cv, f'{name}.csv'), index=False)
        else:
            print(f'Skipping saving {name}.csv as it is empty')

    # Confusion matrix
    dir_path_cm_cv = "C:/Users/filip/Desktop/Analysis_Fungi_Conservation/dowload_variables/NN_model/models/results/final_models_report/confusion_matrix"
    os.makedirs(dir_path_cm_cv, exist_ok=True)

    labels_classification_5 = ['LC', 'NT', 'VU', 'EN', 'CR']
    labels_classification_2 = ['Not Threatened', 'Possibly Threatened']

    #Best test accuracy
    for model_type in ['classification', 'regression']:
        for num_classes in [5, 2]:
            for oversampled in ['yes', 'no']:
                for accuracy in ['balanced', 'normal']:
                    labels = labels_classification_5 if num_classes == 5 else labels_classification_2
                    filename = f'{model_type}_{num_classes}_classes_{accuracy}_{"oversampled" if oversampled == "yes" else "non_oversampled"}_confusion_matrix_best_accuracy.png'
                    save_path = os.path.join(dir_path_cm_cv, filename)
                    visualize_confusion_matrix_highest_test_accuracy(
                        all_models_result, labels, model_type, num_classes, oversampled, accuracy,
                        save_path=save_path, save=True, show=False
                    )

    # Best test balanced accuracy
    for model_type in ['classification', 'regression']:
       for num_classes in [5, 2]:
            for oversampled in ['yes', 'no']:
                for accuracy in ['balanced', 'normal']:
                    labels = labels_classification_5 if num_classes == 5 else labels_classification_2
                    filename = f'{model_type}_{num_classes}_classes_{accuracy}_{"oversampled" if oversampled == "yes" else "non_oversampled"}_confusion_matrix_best_balanced_acc.png'
                    save_path = os.path.join(dir_path_cm_cv, filename)
                    visualize_confusion_matrix_highest_F1_score(
                        all_models_result, labels, model_type, num_classes, oversampled, accuracy,
                        save_path=save_path, save=True, show=False
                    )

    # Best test MCC
    for model_type in ['classification', 'regression']:
        for num_classes in [5, 2]:
            for oversampled in ['yes', 'no']:
                for accuracy in ['balanced', 'normal']:
                    labels = labels_classification_5 if num_classes == 5 else labels_classification_2
                    filename = f'{model_type}_{num_classes}_classes_{accuracy}_{"oversampled" if oversampled == "yes" else "non_oversampled"}_confusion_matrix_best_MCC_score.png'
                    save_path = os.path.join(dir_path_cm_cv, filename)
                    visualize_confusion_matrix_highest_MCC_score(
                        all_models_result, labels, model_type, num_classes, oversampled, accuracy,
                        save_path=save_path, save=True, show=False
                    )
    #endregion


