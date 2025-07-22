import numpy as np  # Import numpy for numerical operations
import pandas as pd  # Import pandas for data manipulation
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Import preprocessing tools
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest classifier
from sklearn.neighbors import KNeighborsClassifier  # Import kNN classifier
from sklearn.metrics import accuracy_score, adjusted_rand_score  # Import metrics for evaluation
from sklearn.cluster import KMeans, AgglomerativeClustering  # Import clustering algorithms

def load_and_preprocess():  # Function to load and preprocess the dataset
    df = pd.read_csv('dermatology.csv', sep='\t')  # Read the CSV file with tab separator
    df.replace('?', np.nan, inplace=True)  # Replace '?' with NaN for missing values
    df.dropna(inplace=True)  # Drop rows with missing values
    df = df.reset_index(drop=True)  # Reset the index after dropping rows
    df.columns = [  # Set the correct column names
        'Erythema', 'Scaling', 'Definite Borders', 'Itching', 'Koebner', 'Polygonal',
        'Follicular', 'Oral', 'Knee', 'Scalp', 'Family History', 'Melanin', 'Eosinophils',
        'PNL', 'Fibrosis', 'Exocytosis', 'Acanthosis', 'Hyperkeratosis', 'Parakeratosis',
        'Clubbing', 'Elongation', 'Thinning', 'Spongiform', 'Munro', 'Focal', 'Disappearance',
        'Vacuolisation', 'Spongiosis', 'Retes', 'Follicular Horn', 'Perifollicular',
        'Inflammatory', 'Band-like', 'Age', 'Disease'
    ]
    df = df.apply(pd.to_numeric)  # Convert all columns to numeric types
    le = LabelEncoder()  # Create a label encoder for the disease column
    df['Disease'] = le.fit_transform(df['Disease'])  # Encode disease names as integers
    print(df)  # Display the entire dataframe after preprocessing
    clinical_cols = [  # List of clinical attribute column names
        'Erythema', 'Scaling', 'Definite Borders', 'Itching', 'Koebner', 'Polygonal',
        'Follicular', 'Oral', 'Knee', 'Scalp', 'Family History', 'Age'
    ]
    histopath_cols = [  # List of histopathological attribute column names
        'Melanin', 'Eosinophils', 'PNL', 'Fibrosis', 'Exocytosis', 'Acanthosis',
        'Hyperkeratosis', 'Parakeratosis', 'Clubbing', 'Elongation', 'Thinning',
        'Spongiform', 'Munro', 'Focal', 'Disappearance', 'Vacuolisation', 'Spongiosis',
        'Retes', 'Follicular Horn', 'Perifollicular', 'Inflammatory', 'Band-like'
    ]
    all_feature_cols = clinical_cols[:-1] + histopath_cols + ['Age']  # All features except Disease
    X_clinical = df[clinical_cols].values  # Extract clinical features as numpy array
    X_histopath = df[histopath_cols].values  # Extract histopathological features as numpy array
    X_all = df[all_feature_cols].values  # Extract all features as numpy array
    y = df['Disease'].values  # Extract target variable as numpy array
    scaler = StandardScaler()  # Create a standard scaler for normalization
    X_clinical_scaled = scaler.fit_transform(X_clinical)  # Standardize clinical features
    X_histopath_scaled = scaler.fit_transform(X_histopath)  # Standardize histopathological features
    X_all_scaled = scaler.fit_transform(X_all)  # Standardize all features
    X_age = df[['Age']].values  # Extract Age as a feature
    return X_age, X_all, X_clinical, X_histopath, X_clinical_scaled, X_histopath_scaled, X_all_scaled, y, le  # Return all processed data

def model1_gradient_descent(X_age, y, le, patient_age=None):  # Model 1: Softmax regression using Age only
    def gradient_descent(X, y, lr=0.01, epochs=1000):  # Inner function for gradient descent
        m = len(y)  # Number of samples
        X_b = np.c_[np.ones((m, 1)), X]  # Add bias (intercept) term to features
        theta = np.zeros((X_b.shape[1], len(np.unique(y))))  # Initialize weights to zero
        y_onehot = np.eye(len(np.unique(y)))[y]  # One-hot encode the labels
        for epoch in range(epochs):  # Iterate for a number of epochs
            logits = X_b @ theta  # Compute logits (linear combination)
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Exponentiate logits for stability
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # Softmax probabilities
            grad = X_b.T @ (probs - y_onehot) / m  # Compute gradient
            theta -= lr * grad  # Update weights using learning rate
        return theta  # Return learned weights
    def predict_gd(X, theta):  # Inner function to predict using learned weights
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term to input
        logits = X_b @ theta  # Compute logits
        return np.argmax(logits, axis=1)  # Return class with highest score
    theta = gradient_descent(X_age, y, lr=0.05, epochs=2000)  # Train the model
    y_pred_gd = predict_gd(X_age, theta)  # Predict on training data
    acc = accuracy_score(y, y_pred_gd)  # Calculate accuracy
    print("Model 1 (GD, Age only) accuracy:", acc)  # Print accuracy
    print("First 5 classified disease types (GD, Age only):", le.inverse_transform(y_pred_gd[:5]))  # Print first 5 predictions
    if patient_age is not None:  # If a patient age is provided
        age_arr = np.array([[patient_age]])  # Create array for the age
        pred_class = predict_gd(age_arr, theta)[0]  # Predict class for the age
        disease_name = le.inverse_transform([pred_class])[0]  # Decode class to disease name
        print(f"Predicted disease type for patient age {patient_age}: {disease_name}")  # Print prediction
        return acc, disease_name  # Return accuracy and prediction
    return acc, None  # Return accuracy only if no age provided

def model2_random_forest(X_clinical, X_histopath, y, le):  # Model 2: Random Forest on clinical and histopathological features
    rf_clin = RandomForestClassifier(n_estimators=100, random_state=42)  # Create Random Forest for clinical
    rf_clin.fit(X_clinical, y)  # Train on clinical features
    y_pred_rf_clin = rf_clin.predict(X_clinical)  # Predict on clinical features
    acc_clin = accuracy_score(y, y_pred_rf_clin)  # Calculate accuracy
    print("Model 2a (Random Forest, clinical attributes) accuracy:", acc_clin)  # Print accuracy
    print("First 5 classified disease types (Random Forest, clinical):", le.inverse_transform(y_pred_rf_clin[:5]))  # Print first 5 predictions

    rf_hist = RandomForestClassifier(n_estimators=100, random_state=42)  # Create Random Forest for histopathological
    rf_hist.fit(X_histopath, y)  # Train on histopathological features
    y_pred_rf_hist = rf_hist.predict(X_histopath)  # Predict on histopathological features
    acc_hist = accuracy_score(y, y_pred_rf_hist)  # Calculate accuracy
    print("Model 2b (Random Forest, histopathological attributes) accuracy:", acc_hist)  # Print accuracy
    print("First 5 classified disease types (Random Forest, histopathological):", le.inverse_transform(y_pred_rf_hist[:5]))  # Print first 5 predictions
    return acc_clin, acc_hist  # Return both accuracies

def model3_knn(X_clinical_scaled, X_histopath_scaled, y, le):  # Model 3: kNN on clinical and histopathological features
    knn_clin = KNeighborsClassifier(n_neighbors=5)  # Create kNN for clinical
    knn_clin.fit(X_clinical_scaled, y)  # Train on clinical features
    y_pred_knn_clin = knn_clin.predict(X_clinical_scaled)  # Predict on clinical features
    acc_clin = accuracy_score(y, y_pred_knn_clin)  # Calculate accuracy
    print("Model 3a (kNN, clinical attributes) accuracy:", acc_clin)  # Print accuracy
    print("First 5 classified disease types (kNN, clinical):", le.inverse_transform(y_pred_knn_clin[:5]))  # Print first 5 predictions

    knn_hist = KNeighborsClassifier(n_neighbors=5)  # Create kNN for histopathological
    knn_hist.fit(X_histopath_scaled, y)  # Train on histopathological features
    y_pred_knn_hist = knn_hist.predict(X_histopath_scaled)  # Predict on histopathological features
    acc_hist = accuracy_score(y, y_pred_knn_hist)  # Calculate accuracy
    print("Model 3b (kNN, histopathological attributes) accuracy:", acc_hist)  # Print accuracy
    print("First 5 classified disease types (kNN, histopathological):", le.inverse_transform(y_pred_knn_hist[:5]))  # Print first 5 predictions
    return acc_clin, acc_hist  # Return both accuracies

def model4_kmeans(X_all_scaled, y, le):  # Model 4: KMeans clustering on all features
    kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42, n_init=10)  # Create KMeans with number of clusters = number of classes
    clusters_kmeans = kmeans.fit_predict(X_all_scaled)  # Fit and predict clusters
    ari = adjusted_rand_score(y, clusters_kmeans)  # Calculate Adjusted Rand Index
    print("Model 4 (KMeans clustering) ARI:", ari)  # Print ARI
    print("Contingency table (KMeans clusters vs. true disease type):")  # Print header for contingency table
    ct = pd.crosstab(clusters_kmeans, le.inverse_transform(y))  # Create contingency table
    print(ct.head(10))  # Print first 10 rows of the table
    return ari  # Return ARI

def model5_agglomerative(X_all_scaled, y, le):  # Model 5: Agglomerative clustering on all features
    agg = AgglomerativeClustering(n_clusters=len(np.unique(y)))  # Create Agglomerative Clustering
    clusters_agg = agg.fit_predict(X_all_scaled)  # Fit and predict clusters
    ari = adjusted_rand_score(y, clusters_agg)  # Calculate Adjusted Rand Index
    print("Model 5 (Agglomerative clustering) ARI:", ari)  # Print ARI
    print("Contingency table (Agglomerative clusters vs. true disease type):")  # Print header for contingency table
    ct = pd.crosstab(clusters_agg, le.inverse_transform(y))  # Create contingency table
    print(ct.head(10))  # Print first 10 rows of the table
    return ari  # Return ARI

def main():  # Main function to run all models
    X_age, X_all, X_clinical, X_histopath, X_clinical_scaled, X_histopath_scaled, X_all_scaled, y, le = load_and_preprocess()  # Load and preprocess data
    model1_gradient_descent(X_age, y, le, patient_age=30)  # Run Model 1 and predict for age 30
    model2_random_forest(X_clinical, X_histopath, y, le)  # Run Model 2
    model3_knn(X_clinical_scaled, X_histopath_scaled, y, le)  # Run Model 3
    model4_kmeans(X_all_scaled, y, le)  # Run Model 4
    model5_agglomerative(X_all_scaled, y, le)  # Run Model 5

if __name__ == "__main__":  # If this script is run directly
    main()  # Call the main function
    