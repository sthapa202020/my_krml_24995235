 

def plot_highly_correlated_matrix(data, lower_threshold=-0.75, upper_threshold=0.75):

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    # Select numerical features
    numerical_features = data.select_dtypes(include=[np.number])
    
    
    corr_matrix = numerical_features.corr()

    # Create a matrix to store filtered correlations
    filtered_corr_matrix = np.zeros_like(corr_matrix)
    
    # a way to hide the correlation that does not meet the criteria using mask
    mask = np.ones_like(corr_matrix, dtype=bool)
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            corr_value = corr_matrix.iloc[i, j]
            if corr_value < lower_threshold or corr_value > upper_threshold:
                filtered_corr_matrix[i, j] = corr_value
                filtered_corr_matrix[j, i] = corr_value  # Symmetric matrix
                mask[i, j] = False
                mask[j, i] = False  # Symmetric mask

    # Converting my filtered correlation matrix to a DataFrame
    filtered_corr_df = pd.DataFrame(filtered_corr_matrix, index=corr_matrix.columns, columns=corr_matrix.columns)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(filtered_corr_df, annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1, mask=mask)
    plt.title('Filtered Correlation Matrix')
    plt.show()
