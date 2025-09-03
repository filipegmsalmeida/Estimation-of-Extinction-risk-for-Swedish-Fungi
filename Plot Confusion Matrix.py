import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import numpy as np
import pandas as pd
import copy
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, balanced_accuracy_score


#region Functions
def load_model_results(INPUT_PATH, MODELS):
    results = {}
    for model_file in MODELS:
        file_path = os.path.join(INPUT_PATH, model_file + '.pkl')
        with open(file_path, "rb") as pickle_file:
            results[model_file] = pickle.load(pickle_file)
    return results

def visualize_confusion_matrix_highest_mean_val_metric(all_model_results, labels, num_classes, oversampling, weight,
                                                accuracy):
    # Filter the results based on provided parameters
    model_results_filtered = [result for result in all_model_results if result['config']['num_classes'] == num_classes and result['config']['oversampling'] == oversampling and result['config']['weight'] == weight and result['config']['accuracy'] == accuracy]

    if not model_results_filtered:
        print(f"No models with {num_classes} classes and oversampling = {oversampling} found.")
        return None, None, None, None, None

    # Select the best model based on accuracy type
    if accuracy == 'balanced':
        best_model_result = max(model_results_filtered, key=lambda x: x['mean_results']['mean_val_metric'])
        mean_balanced_acc = best_model_result['mean_results']['mean_balanced_acc']
        mean_test_accuracy = best_model_result['mean_results']['mean_test_accuracy']
    elif accuracy == 'normal':
        best_model_result = max(model_results_filtered, key=lambda x: x['mean_results']['mean_val_metric'])
        mean_balanced_acc = best_model_result['mean_results']['mean_balanced_acc'] # Not used
        mean_test_accuracy = best_model_result['mean_results']['mean_test_accuracy']
    else:
        raise ValueError("Unsupported accuracy type or model type")

    best_model_settings = best_model_result['config']
    best_model_name = f"{best_model_settings['n_layers']}_{best_model_settings['dropout_rate']}_{best_model_settings['batch_size']}_{best_model_settings['oversampling']}"

    # Extract and process confusion matrices
    confusion_matrices = [fold_result['confusion_matrix'] for fold_result in best_model_result['fold_results']]
    max_shape = max(cm.shape for cm in confusion_matrices)
    padded_confusion_matrices = [
        np.pad(cm, [(0, max_shape[0] - cm.shape[0]), (0, max_shape[1] - cm.shape[1])], mode='constant') for cm in
        confusion_matrices]
    combined_confusion_matrix = np.sum(padded_confusion_matrices, axis=0)
    row_sums = combined_confusion_matrix.sum(axis=1, keepdims=True)
    combined_confusion_matrix_percent = combined_confusion_matrix / row_sums

    return combined_confusion_matrix, combined_confusion_matrix_percent, best_model_name, mean_balanced_acc, mean_test_accuracy


def plot_combined_confusion_matrices(results_list, labels_list, weight_list, num_classes_list, oversampling_list,
                                     accuracy_list, save_path):
    fig, axes = plt.subplots(3, 2, figsize=(25, 18))
    sns.set(font_scale=1.2)
    sns.set_palette("colorblind")
    axes = axes.reshape(3, 2)  # row = condition, col = accuracy

    # Labels for the subplots
    subplot_labels = ["A)", "B)", "C)", "D)", "E)", "F)"]

    row_labels = ["Baseline", "Weight", "Oversampling"]
    col_labels = ["Regular Accuracy", "Balanced Accuracy"]

    for i, (results, labels, num_classes, oversampling, weight, accuracy) in enumerate(
            zip(results_list, labels_list, num_classes_list, oversampling_list, weight_list, accuracy_list)):

        row = i // 2  # 0=baseline, 1=oversampling, 2=weight
        col = i % 2  # 0=regular, 1=balanced
        ax = axes[row, col]

        combined_confusion_matrix, combined_confusion_matrix_percent, best_model_name, mean_balanced_acc, mean_test_accuracy = visualize_confusion_matrix_highest_mean_val_metric(
            results, labels, num_classes, oversampling, weight, accuracy)

        if combined_confusion_matrix is not None:
            # Determine the accuracy type and value
            accuracy_type = "Balanced Accuracy" if accuracy == "balanced" else "Regular Accuracy"
            balanced_accuracy_value = mean_balanced_acc
            regular_accuracy_value = mean_test_accuracy

            # Create the heatmap
            sns.heatmap(combined_confusion_matrix_percent, annot=combined_confusion_matrix, fmt="d", cmap="Blues",
                        cbar=True,
                        xticklabels=labels,
                        yticklabels=labels,
                        # Show labels only for non-oversampled (weighted) models
                        annot_kws={"size": 16, 'weight': 'bold'},
                        linewidths=1, linecolor='gray', ax=ax, vmin=0, vmax=1)

            # Add subplot labels (A), B), C), D)) in the top-left corner
            ax.text(-0.05, 1.15, subplot_labels[i], transform=ax.transAxes,
                         fontsize=22, fontweight='bold', va='top', ha='right')

            # Titles for columns
            if row == 0:
                # First line (bold)
                ax.set_title(col_labels[col], fontsize=22, fontweight='bold', pad=28)

                # Second line (normal) slightly below the first
                ax.text(0.5, 1.02,
                        f"Reg Acc: {regular_accuracy_value:.2f}  Bal Acc: {balanced_accuracy_value:.2f}",
                        transform=ax.transAxes, fontsize=18, fontweight='normal', ha='center')
            else:
                ax.set_title(f"Reg Acc: {regular_accuracy_value:.2f}  Bal Acc: {balanced_accuracy_value:.2f}",
                             fontsize=18)

            # Row labels on the left side
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=22, weight="bold", labelpad=10)


    plt.tight_layout(rect=[0.1, 0.1, 0.9, 0.9])
    plt.savefig(save_path, dpi=300)
    plt.show()


def metrics_from_confusion(cm):
    """
    Compute metrics from a 2x2 confusion matrix.
    Returns: dict with accuracy, f1, mcc, balanced_acc
    """
    if cm.shape != (2, 2):
        raise ValueError(f"Expected 2x2 confusion matrix, got shape {cm.shape}")

    tn, fp, fn, tp = cm.ravel()

    # rebuild true and predicted labels
    y_true = np.array([0] * (tn + fp) + [1] * (fn + tp))
    y_pred = np.array([0] * tn + [1] * fp + [0] * fn + [1] * tp)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred)
    }

def plot_confusion_matrix_class_approach(results, approach, OUTPUT_PATH):
    if approach == '5':

        labels_classification = ['LC', 'NT', 'VU', 'EN', 'CR']
        num_classes_list = [5] * 6

        # For 5-class case
        results_list = [
            results['normal_accuracy_5'],  # baseline normal
            results['balanced_accuracy_5'],  # baseline balanced
            results['normal_accuracy_5'],  # weight normal
            results['balanced_accuracy_5'],  # weight balanced
            results['normal_accuracy_5'],  # oversampling normal
            results['balanced_accuracy_5']  # oversampling balanced
        ]

        outdir = os.path.join(OUTPUT_PATH, 'combined_confusion_matrices_5.png')

    elif approach == '2':

        labels_classification = ['Not Threatened', 'Possibly Threatened']
        num_classes_list = [2] * 6

        # For 2-class case
        results_list = [
            results['normal_accuracy_2'],  # baseline normal
            results['balanced_accuracy_2'],  # oversampling normal
            results['normal_accuracy_2'],  # weight normal
            results['balanced_accuracy_2'],  # baseline balanced
            results['normal_accuracy_2'],  # oversampling balanced
            results['balanced_accuracy_2']  # weight balanced
        ]

        outdir = os.path.join(OUTPUT_PATH, 'combined_confusion_matrices_2.png')

    elif approach == '5_to_2':

        labels_classification = ['Not Threatened', 'Possibly Threatened']
        num_classes_list = [2] * 6

        # Take 5-class results and convert them
        results_list = [
            results['normal_accuracy_5_to_2'],  # baseline normal
            results['balanced_accuracy_5_to_2'],  # oversampling normal
            results['normal_accuracy_5_to_2'],  # weight normal
            results['balanced_accuracy_5_to_2'],  # baseline balanced
            results['normal_accuracy_5_to_2'],  # oversampling balanced
            results['balanced_accuracy_5_to_2']  # weight balanced
        ]

        outdir = os.path.join(OUTPUT_PATH, 'combined_confusion_matrices_5_to_2.png')

    labels_list = [labels_classification] * 6

    oversampling_list = ['no', 'no', 'no', 'no', 'yes', 'yes']
    weight_list = ['no', 'no', 'yes', 'yes', 'no', 'no']
    accuracy_list = ['normal', 'balanced', 'normal', 'balanced', 'normal', 'balanced']

    plot_combined_confusion_matrices(
        results_list, labels_list, weight_list, num_classes_list, oversampling_list, accuracy_list,
        outdir
    )

def convert_results_5_to_2_new(results, keys):
    """
    Convert selected keys from 5-class to 2-class and recalc metrics.
    keys: list of keys to convert, e.g., ['normal_accuracy_5', 'balanced_accuracy_5']
    """
    class_map_5_to_2 = {
        'LC': 'Not Threatened',
        'NT': 'Not Threatened',
        'VU': 'Possibly Threatened',
        'EN': 'Possibly Threatened',
        'CR': 'Possibly Threatened'
    }

    for key in keys:
        for run in results[key]:
            for fold in run['fold_results']:
                cm_5 = fold['confusion_matrix']
                if cm_5.shape == (5, 5):
                    cm_2 = np.zeros((2, 2), dtype=int)
                    labels_5 = ['LC', 'NT', 'VU', 'EN', 'CR']
                    label_to_index_2 = {'Not Threatened': 0, 'Possibly Threatened': 1}

                    for r5, true_label in enumerate(labels_5):
                        for c5, pred_label in enumerate(labels_5):
                            r2 = label_to_index_2[class_map_5_to_2[true_label]]
                            c2 = label_to_index_2[class_map_5_to_2[pred_label]]
                            cm_2[r2, c2] += cm_5[r5, c5]

                    fold['confusion_matrix'] = cm_2

                    # Recompute metrics
                    metrics = metrics_from_confusion(cm_2)
                    fold['test_accuracy'] = metrics["accuracy"]
                    fold['balanced_acc'] = metrics["balanced_acc"]
                    fold['f1_score'] = metrics["f1"]
                    fold['mcc_test'] = metrics["mcc"]


            # Update mean metrics for the run without touching mean_val_metric
            run['mean_results']['mean_test_accuracy'] = np.mean([fold['test_accuracy'] for fold in run['fold_results']])
            run['mean_results']['mean_balanced_acc'] = np.mean([fold['balanced_acc'] for fold in run['fold_results']])
            run['mean_results']['mean_f1'] = np.mean([fold['f1_score'] for fold in run['fold_results']])
            run['mean_results']['mean_mcc'] = np.mean([fold['mcc_test'] for fold in run['fold_results']])
            run['config']['num_classes'] = 2


    return results


def save_csv_convertion_5_to_2(results, keys, OUTPUT_PATH):
    """
    Save mean metrics for each results key to a separate CSV file.

    results: dict containing results lists (e.g., normal_accuracy_5_to_2)
    keys: list of keys to save (e.g., ['normal_accuracy_5_to_2', 'balanced_accuracy_5_to_2'])
    OUTPUT_PATH: folder to save CSVs
    """
    for key in keys:
        rows = []
        for run in results[key]:
            config = run['config']

            mean_best_epoch = np.mean([fold['best_epoch'] for fold in run['fold_results']])
            mean_results = run.get('mean_results', {})

            row = {
                'num_classes': config.get('num_classes', np.nan),
                'accuracy': config.get('accuracy', ''),
                'n_layers': config.get('n_layers', ''),
                'dropout_rate': config.get('dropout_rate', ''),
                'batch_size': config.get('batch_size', ''),
                'oversampling': config.get('oversampling', ''),
                'weight': config.get('weight', ''),
                'mean_val_metric': mean_results.get('mean_val_metric', np.nan),
                'mean_best_epoch': mean_best_epoch,
                'mean_f1_score': mean_results.get('mean_f1', np.nan),
                'mean_test_accuracy': mean_results.get('mean_test_accuracy', np.nan),
                'mean_mcc_test': mean_results.get('mean_mcc', np.nan),
                'mean_balanced_acc': mean_results.get('mean_balanced_acc', np.nan)
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_file = os.path.join(OUTPUT_PATH, f"{key}.csv")
        df.to_csv(csv_file, index=False)
        print(f"Saved CSV: {csv_file}")


def plot_best_confusion_matrix(results, OUTPUT_PATH):
    labels_classification = ['Not Threatened', 'Possibly Threatened']
    num_classes_list = 2

    # Take 5-class results and convert them
    results_list = results['balanced_accuracy_5_to_2']


    labels_list = [labels_classification]

    oversampling_list = 'yes'
    weight_list = 'no'
    accuracy_list = 'balanced'


    sns.set(font_scale=1.2)
    sns.set_palette("colorblind")

    # Get the best confusion matrix and metrics
    combined_confusion_matrix, combined_confusion_matrix_percent, best_model_name, mean_balanced_acc, mean_test_accuracy = visualize_confusion_matrix_highest_mean_val_metric(
        results_list, labels_list, num_classes_list, oversampling_list, weight_list, accuracy_list)

    if combined_confusion_matrix is not None:
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            combined_confusion_matrix_percent,
            annot=combined_confusion_matrix,
            fmt="d",
            cmap="Blues",
            cbar=True,
            xticklabels=labels_classification,
            yticklabels=labels_classification,
            annot_kws={"size": 16, 'weight': 'bold'},
            linewidths=1,
            linecolor='gray',
            vmin=0,
            vmax=1,
            ax=ax
        )

        # Title with accuracy info
        ax.set_title(
            f" Reg Acc: {mean_test_accuracy:.2f}, Bal Acc: {mean_balanced_acc:.2f}",
            fontsize=18
        )

        ax.set_xlabel("Predicted Labels", fontsize=14)
        ax.set_ylabel("True Labels", fontsize=14)

        plt.tight_layout()
        outdir = os.path.join(OUTPUT_PATH, 'best_confusion_matrices_5_to_2.png')
        plt.savefig(outdir, dpi=300)
        plt.show()
#endregion

def main(INPUT_PATH, OUTPUT_PATH, MODELS, PATH_5_TO_2):
    results = load_model_results(INPUT_PATH, MODELS)

    plot_confusion_matrix_class_approach(results, approach='5', OUTPUT_PATH=OUTPUT_PATH)
    plot_confusion_matrix_class_approach(results, approach='2', OUTPUT_PATH=OUTPUT_PATH)

    # Create new lists inside results
    results['normal_accuracy_5_to_2'] = copy.deepcopy(results['normal_accuracy_5'])
    results['balanced_accuracy_5_to_2'] = copy.deepcopy(results['balanced_accuracy_5'])

    # Convert and recalc metrics
    results = convert_results_5_to_2_new(results, ['normal_accuracy_5_to_2', 'balanced_accuracy_5_to_2'])

    # Save to CSV
    save_csv_convertion_5_to_2(results, ['normal_accuracy_5_to_2', 'balanced_accuracy_5_to_2'], PATH_5_TO_2)

    # Plot confusion matrices for 5_to_2
    plot_confusion_matrix_class_approach(results, approach='5_to_2', OUTPUT_PATH=OUTPUT_PATH)

    plot_best_confusion_matrix(results, OUTPUT_PATH=OUTPUT_PATH)

# --- Run the main function ---
if __name__ == '__main__':
    INPUT_PATH = 'results/save_model/'
    OUTPUT_PATH = 'results/final_plots/'
    PATH_5_TO_2 = ('results/tables/')

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    MODELS = ['balanced_accuracy_2', 'balanced_accuracy_5', 'normal_accuracy_2', 'normal_accuracy_5']

    main(INPUT_PATH, OUTPUT_PATH, MODELS, PATH_5_TO_2)

