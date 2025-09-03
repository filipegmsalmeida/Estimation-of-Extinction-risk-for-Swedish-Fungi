import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score
import os
import glob
from sklearn.metrics import balanced_accuracy_score

#region Functions
def plot_nocc_against_AOO_and_EOO(df, outdir):
    # plotting n_occ against AOO and EOO
    plt.figure(figsize=(10,5))
    plt.scatter(df['n_occ'], df['AOO2km'], alpha=0.5, label='AOO')
    plt.scatter(df['n_occ'], df['EOOkm2'], alpha=0.5, label='EOO')
    plt.xlabel('Number of Occurrences')
    plt.ylabel('Area (AOO in 2km, EOO in km²)')
    plt.legend()
    plt.savefig(os.path.join(outdir, 'n_occ_vs_AOO_EOO.png'))
    plt.show()


def calculate_mean_accuracy_both(df, predictions_all, n_groups=5):
    df_filtered = df.dropna(subset=['RedListCategory'])
    metrics_list = []

    for fold_name, fold_data in predictions_all.groupby('fold'):
        df_merged = pd.merge(fold_data, df_filtered[['species', 'n_occ', 'EOOkm2', 'AOO2km']], on='species',
                             how='inner')

        # Occurrence densities
        df_merged['occ/EOO'] = df_merged['n_occ'] / df_merged['EOOkm2']
        df_merged['occ/AOO'] = df_merged['n_occ'] / df_merged['AOO2km']

        # Group by quantiles
        df_merged['density_group'] = pd.qcut(df_merged['occ/EOO'], q=n_groups, labels=False)
        for group in sorted(df_merged['density_group'].unique()):
            group_data = df_merged[df_merged['density_group'] == group]
            acc = accuracy_score(group_data['y_true'], group_data['y_pred'])
            metrics_list.append(
                {'fold': fold_name, 'group': f'Group {group + 1}', 'metric': 'occ/EOO', 'accuracy': acc})

        df_merged['density_group'] = pd.qcut(df_merged['occ/AOO'], q=n_groups, labels=False)
        for group in sorted(df_merged['density_group'].unique()):
            group_data = df_merged[df_merged['density_group'] == group]
            acc = accuracy_score(group_data['y_true'], group_data['y_pred'])
            metrics_list.append(
                {'fold': fold_name, 'group': f'Group {group + 1}', 'metric': 'occ/AOO', 'accuracy': acc})

    metrics_df = pd.DataFrame(metrics_list)
    mean_metrics_df = metrics_df.groupby(['group', 'metric'])['accuracy'].mean().reset_index()
    return mean_metrics_df


def calculate_mean_balanced_accuracy(df, predictions_all, n_groups=10):
    df_filtered = df.dropna(subset=['RedListCategory'])
    metrics_list = []

    for fold_name, fold_data in predictions_all.groupby('fold'):
        df_merged = pd.merge(fold_data, df_filtered[['species', 'n_occ', 'AOO2km', 'EOOkm2']], on='species', how='inner')

        # Calculate occurrence densities
        for metric, area_col in zip(['EOO', 'AOO'], ['EOOkm2', 'AOO2km']):
            df_merged[f'occ/{metric}'] = df_merged['n_occ'] / df_merged[area_col]
            df_merged['density_group'] = pd.qcut(df_merged[f'occ/{metric}'], q=n_groups, labels=False)

            for group in sorted(df_merged['density_group'].unique()):
                group_data = df_merged[df_merged['density_group'] == group]
                acc = balanced_accuracy_score(group_data['y_true'], group_data['y_pred'])
                metrics_list.append({
                    'fold': fold_name,
                    'group': f'Group {group + 1}',
                    'metric': metric,
                    'balanced_accuracy': acc
                })

    metrics_df = pd.DataFrame(metrics_list)
    # Average across folds
    mean_metrics_df = metrics_df.groupby(['group', 'metric'])['balanced_accuracy'].mean().reset_index()
    return mean_metrics_df

def plot_balanced_accuracy_vs_both(metrics_df, OUTDIR):
    plt.figure(figsize=(12,6))
    sns.barplot(data=metrics_df, x='group', y='balanced_accuracy', hue='metric', palette='viridis')
    plt.xticks(rotation=45)
    plt.xlabel('Sampling Density Group (occ/EOO or occ/AOO range)')
    plt.ylabel('Balanced Accuracy')
    plt.title('Model Balanced Accuracy vs Sampling Density')
    plt.legend(title='Metric')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'balanced_accuracy_vs_density.png'))
    plt.show()


def plot_accuracy_vs_both(metrics_df, OUTDIR):
    plt.figure(figsize=(12,6))
    sns.barplot(data=metrics_df, x='group', y='accuracy', hue='metric', palette='viridis')
    plt.xticks(rotation=45)
    plt.xlabel('Sampling Density Group')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy vs Sampling Density per EOO and AOO')
    plt.ylim(0, 1)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'accuracy_vs_density_EOO_AOO.png'))
    plt.show()

def combined_plots(df, metrics_df, outdir):
    # Ensure group column matches the case used in metrics generation
    metrics_df['group'] = metrics_df['group'].str.strip()  # remove extra spaces if any

    # Define the exact order of groups
    group_order = [f'Group {i}' for i in range(1, 11)]
    # Keep only existing groups to avoid NaNs
    group_order_filtered = [g for g in group_order if g in metrics_df['group'].values]
    metrics_df['group'] = pd.Categorical(metrics_df['group'], categories=group_order_filtered, ordered=True)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))  # 1 row, 2 columns

    # -------------------------------
    # A) n_occ vs AOO and EOO
    axes[0].scatter(df['n_occ'], df['AOO2km'], alpha=0.5, label='AOO')
    axes[0].scatter(df['n_occ'], df['EOOkm2'], alpha=0.5, label='EOO')
    axes[0].set_xlabel('Number of Occurrences', fontsize=14)
    axes[0].set_ylabel('Area (AOO in 2km², EOO in km²)', fontsize=14)
    axes[0].tick_params(axis='y', rotation=90, labelsize=12)
    axes[0].tick_params(axis='x', labelsize=12)
    axes[0].legend()
    axes[0].text(-0.05, 1.05, 'A)', transform=axes[0].transAxes,
                 fontsize=16, fontweight='bold', va='top', ha='right')

    # -------------------------------
    # B) Balanced Accuracy vs Sampling Density
    sns.barplot(
        data=metrics_df,
        x='group',
        y='balanced_accuracy',
        hue='metric',
        palette='viridis',
        ax=axes[1]
    )
    axes[1].set_xticklabels(axes[1].get_xticklabels())
    axes[1].set_xlabel('Sampling Density Group (occ/EOO or occ/AOO range)', fontsize=14)
    axes[1].set_ylabel('Balanced Accuracy', fontsize=14)
    axes[0].tick_params(axis='y', rotation=90, labelsize=12)
    axes[0].tick_params(axis='x', labelsize=12)
    axes[1].set_ylim(0, 1)
    axes[1].legend(title='Metric')
    axes[1].text(-0.05, 1.05, 'B)', transform=axes[1].transAxes,
                 fontsize=16, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'combined_bias_plots.png'), dpi=300)
    plt.show()

#endregion

def main(df, predictions_all, OUTDIR):
    # Plot n_occ against AOO and EOO
    plot_nocc_against_AOO_and_EOO(df, OUTDIR)

    # Calculate mean accuracy per density group and plot
    mean_accuracy_df = calculate_mean_accuracy_both(df, predictions_all, n_groups=10)
    mean_accuracy_balanced_df = calculate_mean_balanced_accuracy(df, predictions_all, n_groups=10)

    # Plot accuracy vs density
    plot_accuracy_vs_both(mean_accuracy_df, OUTDIR)
    plot_balanced_accuracy_vs_both(mean_accuracy_balanced_df, OUTDIR)

    combined_plots(df, mean_accuracy_balanced_df, OUTDIR)

if __name__ == '__main__':
    INPUT_PATH = 'data/all_data.csv'
    OUTDIR = 'results/final_plots'
    PRED_MODEL_DIR = 'results/models_and_test_data/'
    fold_files = glob.glob(f"{PRED_MODEL_DIR}/predictions_Oversampled_5_balanced_60_30_0.1_40_fold*.csv")

    # Add a fold column when reading each file
    predictions_all = pd.concat([
        pd.read_csv(f).assign(fold=i + 1)  # i+1 = fold number
        for i, f in enumerate(fold_files)]
    )

    df = pd.read_csv(INPUT_PATH)

    main(df, predictions_all, OUTDIR)



mean_accuracy_df = calculate_mean_accuracy_both(df, predictions_all, n_groups=10)
plot_accuracy_vs_both(mean_accuracy_df, OUTDIR)