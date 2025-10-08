import numpy as np                                      # type: ignore
import pandas as pd                                     # type: ignore
import matplotlib.pyplot as plt                         # type: ignore
import matplotlib.cm as cm                              # type: ignore
import matplotlib.colors as mcolors                     # type: ignore
from matplotlib.patches import Rectangle                # type: ignore
import seaborn as sns                                   # type: ignore
import dcor                                             # type: ignore
from sklearn.preprocessing import StandardScaler        # type: ignore
from sklearn.decomposition import PCA                   # type: ignore
from scipy.stats import gmean                           # type: ignore
from statsmodels.graphics.tsaplots import plot_pacf     # type: ignore

def eval_nans(df: pd.DataFrame, preformat: bool=True) -> str:
    if preformat:
        return_str = '<p>'
    else:
        return_str = '\n'
    
    nulls = df.isna()
    missing_entries = nulls.sum().sum()
    rows_with_missing = nulls.any(axis=1).sum() 
    # nan_counts = nulls.groupby(nulls.ne(nulls.shift()).cumsum()).sum()
    # return_str += f'Consecutive NaNs in each column:{nan_counts}\n'
    return_str += f'Total rows with missing entries: {rows_with_missing}\n'
    return_str += f'Total missing entries in the dataset: {missing_entries}\n'

    total_entries = df.size
    total_rows = df.shape[0]
    percentage_missing = (missing_entries / total_entries) * 100
    percentage_rows_missing = (rows_with_missing / total_rows) * 100
    return_str += f'Percentage of rows with missing entries: {percentage_rows_missing:.2f}%\n'
    return_str += f'Percentage of missing entries: {percentage_missing:.2f}%'

    if preformat:
        return_str = return_str.replace('\n', '<br>')
        return_str += '</p>'
    else:
        return_str = '\n'

    return return_str

def stat_analyze(df):
    df_features = df.copy().select_dtypes(include=np.number).dropna()
    row_labels = ['mean', 'stdev', 'median', 'mode', 'min', 'max', 'gmean', 'variance', 'skewness', 'kurtosis']
    df_stats = pd.DataFrame(index=row_labels, columns=df_features.columns)
    df_stats.index.name = 'statistic'
    df_stats.columns.name = 'feature'

    # Fill the stats dataframe with corresponding features
    df_stats.loc['mean'] = df_features.mean()
    df_stats.loc['stdev'] = df_features.std()
    df_stats.loc['median'] = df_features.median()
    df_stats.loc['mode'] = df_features.mode().iloc[0]
    df_stats.loc['min'] = df_features.min()
    df_stats.loc['max'] = df_features.max()
    df_stats.loc['gmean'] = df_features.apply(gmean)
    df_stats.loc['variance'] = df_features.var()
    df_stats.loc['skewness'] = df_features.skew()
    df_stats.loc['kurtosis'] = df_features.kurt()

    return df_stats

def corrplot(df):
    """
    Generates a pair plot with distance correlation values in the upper triangle,
    KDE plots on the diagonal, and scatter plots in the lower triangle.
    """

    df_features = df.copy().select_dtypes(include=np.number).dropna()
    features = df_features.columns
    n_features = len(features)
    df_scaled = pd.DataFrame(StandardScaler().fit_transform(df_features), columns=features).astype(np.float64)
    
    fig, axs = plt.subplots(n_features, n_features, figsize=(n_features * 1.5, n_features * 1), constrained_layout=True)

    for n, row in enumerate(features):
        for m, col in enumerate(features):
            ax = axs[n, m]

            # Correlation heatmap cells (upper triangle)
            if m > n:
                corr_val = dcor.distance_correlation(df_scaled[row].to_numpy(), df_scaled[col].to_numpy()) # Should return a float in the range [0, 1]

                cmap = cm.get_cmap('Greens')
                color = cmap(corr_val)
                color = cmap(corr_val)

                # Create a colored rectangle with the correlation value
                rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=color, edgecolor='white', linewidth=0.2)
                ax.add_patch(rect)
                ax.text(0.5, 0.5, f'{corr_val:.2f}', ha='center', va='center', color='black', fontsize=12, transform=ax.transAxes)
                ax.axis('off')

            # KDE plots (diagonal)
            elif n == m:
                sns.kdeplot(data=df, x=col, ax=ax, fill=True, color=sns.color_palette('viridis', n_features)[n])
                
            # Scatter plots (lower triangle)
            else: # m < n
                sns.scatterplot(data=df, x=col, y=row, ax=ax, s=5, marker='.', color=sns.color_palette('viridis', n_features)[n])

            # Set x and y labels for the leftmost column and bottom row
            if m == 0:
                ax.set_ylabel(row, fontsize=10, rotation=30, va='center', ha='right', labelpad=10)
            else:
                ax.set_ylabel('')
            if n < n_features - 1:
                ax.set_xlabel('')
            
            # Clear interior ticks
            ax.set_xticks([])
            ax.set_yticks([])

    return fig

def pca_plot(df: pd.DataFrame, var_threshold=0.95):
    # Select only numeric columns for PCA
    df_features = df.select_dtypes(include=np.number).dropna()

    # Standardize the features
    df_scaled = pd.DataFrame(StandardScaler().fit_transform(df_features), columns=df_features.columns).astype(np.float64)

    # PCA and Component Selection
    pca = PCA()
    pca.fit(df_scaled)

    # Determine the number of components needed to reach the variance threshold
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.where(cumulative_variance >= var_threshold)[0][0] + 1

    # Rerun PCA with the selected number of components
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df_scaled)

    # Create a DataFrame for the principal components
    pc_df = pd.DataFrame(data=principal_components, columns=[f'PC_{i+1}' for i in range(n_components)])
    pc_scaled = pd.DataFrame(StandardScaler().fit_transform(pc_df), columns=pc_df.columns).astype(np.float64)

    # Calculate distance correlation between Features and Principal Components
    dist_corr_matrix = pd.DataFrame(index=df_features.columns, columns=pc_df.columns)

    # Calculate distance correlation for each feature-PC pair
    for feature in df_features.columns:
        for pc in pc_df.columns:
            dist_corr = dcor.distance_correlation(df_scaled[feature].to_numpy(), pc_scaled[pc].to_numpy())
            dist_corr_matrix.loc[feature, pc] = dist_corr

    dist_corr_matrix = dist_corr_matrix.astype(float)

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        dist_corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='viridis',
        linewidths=.5,
        ax=ax
    )
    ax.set_title(
        f'Distance Correlation of Features and Principal Components\n(Explaining {var_threshold*100}% of Variance)',
        fontsize=16,
        fontweight='bold'
    )
    ax.set_xlabel('Principal Components', fontsize=12)
    ax.set_ylabel('Original Features', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    return fig

def pacf_plot(df, maxlags=15):
    """
    Generates a PACF plot for each numeric feature in the DataFrame.
    """
    df_features = df.select_dtypes(include=np.number).dropna()
    features = df_features.columns

    fig, axes = plt.subplots(len(features), 1, figsize=(12, 12), sharex=True)

    for i, column in enumerate(features):
        plot_pacf(df_features[column], lags=maxlags, ax=axes[i], title=f'{column}')

    fig.supxlabel('Lag', fontsize=12)
    fig.supylabel('Partial Autocorrelation', fontsize=12)
    fig.tight_layout()

    return fig