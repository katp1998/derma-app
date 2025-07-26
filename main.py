import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ASSUMPTION: as it is not specified, assumtion is made these are the disease names are labelled as integers in the dataset:
disease_map = {
    1: 'psoriasis',
    2: 'seboreic dermatitis',
    3: 'lichen planus',
    4: 'pityriasis rosea',
    5: 'cronic dermatitis',
    6: 'pityriasis rubra pilaris'
}

# Define the columns for clinical and histopathological attributes:
clinical_cols = [ 
    'Erythema', 'Scaling', 'Definite Borders', 'Itching', 'Koebner', 'Polygonal', 'Follicular', 'Oral', 'Knee', 'Scalp', 'Family History', 'Age'
]
histopath_cols = [
    'Melanin', 'Eosinophils', 'PNL', 'Fibrosis', 'Exocytosis', 'Acanthosis', 'Hyperkeratosis', 'Parakeratosis', 'Clubbing', 'Elongation', 'Thinning', 'Spongiform', 'Munro', 'Focal', 'Disappearance', 'Vacuolisation', 'Spongiosis', 'Retes', 'Follicular Horn', 'Perifollicular', 'Inflammatory', 'Band-like'
]

# @desc: ensure LabelEncoder mapping and disease_map are in sync:
def update_disease_mapping(label_encoder):
    disease_number_to_name = {
        1: 'psoriasis',
        2: 'seboreic dermatitis',
        3: 'lichen planus',
        4: 'pityriasis rosea',
        5: 'cronic dermatitis',
        6: 'pityriasis rubra pilaris'
    }
    global disease_map
    # map encoded label to disease name using the number-to-name mapping:
    disease_map = {i: disease_number_to_name[int(disease_number)] for i, disease_number in enumerate(label_encoder.classes_)}

# @desc: prep and load the dataset:
def load_and_preprocess():  
    # Read the CSV file with tab separator:
    df = pd.read_csv('dermatology.csv', sep='\t')  
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)

    # Set the correct column names with proper spellings:
    df.columns = [  
        'Erythema', 'Scaling', 'Definite Borders', 'Itching', 'Koebner', 'Polygonal', 'Follicular', 'Oral', 'Knee', 'Scalp', 'Family History',
        'Melanin', 'Eosinophils', 'PNL', 'Fibrosis', 'Exocytosis', 'Acanthosis', 'Hyperkeratosis', 'Parakeratosis', 'Clubbing', 'Elongation', 'Thinning', 'Spongiform', 'Munro', 'Focal', 'Disappearance', 'Vacuolisation', 'Spongiosis', 'Retes', 'Follicular Horn', 'Perifollicular', 'Inflammatory', 'Band-like', 'Age', 'Disease'
    ]

    df = df.apply(pd.to_numeric)

     # Create a label encoder for the disease column:
    le = LabelEncoder()
    df['Disease'] = le.fit_transform(df['Disease'])
    
    update_disease_mapping(le)
    
    # print(df)

    all_feature_cols = clinical_cols[:-1] + histopath_cols + ['Age']

    X_clinical = df[clinical_cols].values
    X_histopath = df[histopath_cols].values
    X_all = df[all_feature_cols].values

    y = df['Disease'].values 
    # Normalize the features using StandardScaler:
    scaler = StandardScaler() 
    X_clinical_scaled = scaler.fit_transform(X_clinical)
    X_histopath_scaled = scaler.fit_transform(X_histopath)
    X_all_scaled = scaler.fit_transform(X_all)
    X_age = df[['Age']].values
    return X_age, X_all, X_clinical, X_histopath, X_clinical_scaled, X_histopath_scaled, X_all_scaled, y, le

# Determine the type of disease based on the patient’s Age. Use gradient descent (GD) to build your regression model (model 1). 
def model1_gradient_descent(X_age, y, label_encoder=None, new_age=None):
    no_of_patients = X_age.shape[0]
    no_of_diseases = len(np.unique(y))

    # column of 1s for the bias
    X_with_bias = np.c_[np.ones((no_of_patients, 1)), X_age]

    y_onehot = np.eye(no_of_diseases)[y]

    # empty weights for the model
    weights = np.zeros((X_with_bias.shape[1], no_of_diseases))

    learning_rate = 0.05
    epochs = 500
    losses = []

    # gradient descent
    for epoch in range(epochs):
        logits = X_with_bias @ weights

        # softmax function to convert logits to probabilities
        exp_values = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # for stability
        probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        # cross-entropy loss
        loss = -np.sum(y_onehot * np.log(probs + 1e-15)) / no_of_patients
        losses.append(loss)

        gradient = X_with_bias.T @ (probs - y_onehot) / no_of_patients

        weights -= learning_rate * gradient

    # prediction:
    def predict(input_X):
        input_with_bias = np.c_[np.ones((input_X.shape[0], 1)), input_X]
        logits = input_with_bias @ weights
        return np.argmax(logits, axis=1)

    predictions = predict(X_age)
    accuracy = accuracy_score(y, predictions)

    pred_names = [disease_map.get(label, str(label)) for label in predictions]
    true_names = [disease_map.get(label, str(label)) for label in y]

    print('Model accuracy:', accuracy)

    if new_age is not None:
        new_age_input = np.array([[new_age]])
        predicted_class = predict(new_age_input)[0]
        disease_name = disease_map.get(predicted_class, str(predicted_class))
        print(f'Predicted disease for age {new_age}: {disease_name}')
        result = (accuracy, disease_name)
    else:
        result = (accuracy, None)

    # loss function
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label='Loss Function - Gradient Descent')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Function Value')
    plt.title('Loss Function Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Confusion matrix
    unique_names = sorted(set(true_names) | set(pred_names))
    cm = confusion_matrix(true_names, pred_names, labels=unique_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_names, yticklabels=unique_names)
    plt.xlabel('Predicted Disease')
    plt.ylabel('True Disease')
    plt.title('Gradient Descent - Confusion Matrix')
    plt.tight_layout()
    plt.show()

    return result

# @desc: convert symptom text arrays to feature vectors
def symptoms_to_features(symptom_list, feature_names):
    features = np.zeros(len(feature_names))
    for i, name in enumerate(feature_names):
        if name == 'Age':
            age_val = next((s.split(":")[1] for s in symptom_list if s.startswith("Age:")), None)
            if age_val is not None:
                features[i] = float(age_val)
            else:
                features[i] = 0
        else:
            features[i] = 1 if name in symptom_list else 0
    return features.reshape(1, -1)

# Use random forest on the clinical as well as histopathological attributes to classify the disease type (model2). 
def model2_random_forest(clinical_symptoms_text, histopath_symptoms_text, labels, label_encoder):
    X_age, X_all, X_clinical, X_histopath, X_clinical_scaled, X_histopath_scaled, X_all_scaled, y, le = load_and_preprocess()

    n_estimators_list = [10, 20, 30, 50, 75, 100]
    loss_clin = []
    loss_hist = []
    loss_all = []
    
    for n_est in n_estimators_list:
        # Clinical model:
        rf_clin_temp = RandomForestClassifier(n_estimators=n_est, random_state=42)
        rf_clin_temp.fit(X_clinical, y)
        y_pred_clin_temp = rf_clin_temp.predict(X_clinical)
        loss_clin.append(1 - accuracy_score(y, y_pred_clin_temp))
        
        # Histopathological model:
        rf_hist_temp = RandomForestClassifier(n_estimators=n_est, random_state=42)
        rf_hist_temp.fit(X_histopath, y)
        y_pred_hist_temp = rf_hist_temp.predict(X_histopath)
        loss_hist.append(1 - accuracy_score(y, y_pred_hist_temp))
        
        # Combined model:
        rf_all_temp = RandomForestClassifier(n_estimators=n_est, random_state=42)
        rf_all_temp.fit(X_all, y)
        y_pred_all_temp = rf_all_temp.predict(X_all)
        loss_all.append(1 - accuracy_score(y, y_pred_all_temp))

    rf_clin = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clin.fit(X_clinical, y)
    rf_hist = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_hist.fit(X_histopath, y)
    rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_all.fit(X_all, y)

    # accuracy:
    y_pred_clin = rf_clin.predict(X_clinical)
    y_pred_hist = rf_hist.predict(X_histopath)
    y_pred_all = rf_all.predict(X_all)
    
    acc_clin = accuracy_score(y, y_pred_clin)
    acc_hist = accuracy_score(y, y_pred_hist)
    acc_all = accuracy_score(y, y_pred_all)
    
    print(f"Training Accuracy (Clinical): {acc_clin:.4f}")
    print(f"Training Accuracy (Histopathological): {acc_hist:.4f}")
    print(f"Training Accuracy (All features): {acc_all:.4f}")

    clinical_features = symptoms_to_features(clinical_symptoms_text, clinical_cols)
    histopath_features = symptoms_to_features(histopath_symptoms_text, histopath_cols)
    pred_clin = rf_clin.predict(clinical_features)[0]
    pred_hist = rf_hist.predict(histopath_features)[0]

    # remove 'Age' from clinical_features to avoid duplication -- 'Age' is at the end of clinical_cols and all_feature_cols:
    clinical_features_no_age = clinical_features[0, :-1]
    combined_features = np.concatenate([clinical_features_no_age, histopath_features[0], clinical_features[0, -1:]])  # add Age at the end
    combined_features = combined_features.reshape(1, -1)
    pred_combined = rf_all.predict(combined_features)[0]

    disease_clin_name = disease_map[pred_clin]
    disease_hist_name = disease_map[pred_hist]
    disease_combined_name = disease_map[pred_combined]

    print(f"Predicted disease (clinical symptoms): {disease_clin_name}")
    print(f"Predicted disease (histopathological symptoms): {disease_hist_name}")
    print(f"Predicted disease (all symptoms combined): {disease_combined_name}")

    # loss function:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(n_estimators_list, loss_clin, 'b-o', label='Clinical Features', linewidth=2, markersize=6)
    plt.plot(n_estimators_list, loss_hist, 'r-s', label='Histopathological Features', linewidth=2, markersize=6)
    plt.plot(n_estimators_list, loss_all, 'g-^', label='All Features', linewidth=2, markersize=6)
    plt.xlabel('Number of Estimators')
    plt.ylabel('Misclassification Rate (Loss)')
    plt.title('Random Forest Loss Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(max(loss_clin), max(loss_hist), max(loss_all)) * 1.1)

    # confusion matrices:
    plt.subplot(1, 2, 2)
    cm_all = confusion_matrix(y, y_pred_all)
    sns.heatmap(cm_all, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - All Features')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cm_clin = confusion_matrix(y, y_pred_clin)
    sns.heatmap(cm_clin, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix - Clinical Features')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    cm_hist = confusion_matrix(y, y_pred_hist)
    sns.heatmap(cm_hist, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('Confusion Matrix - Histopathological Features')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    sns.heatmap(cm_all, annot=True, fmt='d', cmap='Blues', ax=axes[2])
    axes[2].set_title('Confusion Matrix - All Features')
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()

    return disease_clin_name, disease_hist_name, disease_combined_name

# Use kNN on the clinical attributes and histopathological attributes to classify the disease type and report your accuracy (model3). 
def model3_knn(clinical_symptoms_text, histopath_symptoms_text, labels, label_encoder):
    X_age, X_all, X_clinical, X_histopath, X_clinical_scaled, X_histopath_scaled, X_all_scaled, y, le = load_and_preprocess()

    k_neighbors_list = [1, 3, 5, 7, 9, 11]
    loss_clin = []
    loss_hist = []
    loss_all = []
    
    for k in k_neighbors_list:
        # Clinical model:
        knn_clin_temp = KNeighborsClassifier(n_neighbors=k)
        knn_clin_temp.fit(X_clinical_scaled, y)
        y_pred_clin_temp = knn_clin_temp.predict(X_clinical_scaled)
        loss_clin.append(1 - accuracy_score(y, y_pred_clin_temp))
        
        # Histopathological model:
        knn_hist_temp = KNeighborsClassifier(n_neighbors=k)
        knn_hist_temp.fit(X_histopath_scaled, y)
        y_pred_hist_temp = knn_hist_temp.predict(X_histopath_scaled)
        loss_hist.append(1 - accuracy_score(y, y_pred_hist_temp))
        
        # Combined model:
        knn_all_temp = KNeighborsClassifier(n_neighbors=k)
        knn_all_temp.fit(X_all_scaled, y)
        y_pred_all_temp = knn_all_temp.predict(X_all_scaled)
        loss_all.append(1 - accuracy_score(y, y_pred_all_temp))

    # k=5:
    knn_clin = KNeighborsClassifier(n_neighbors=5)
    knn_clin.fit(X_clinical_scaled, y)
    knn_hist = KNeighborsClassifier(n_neighbors=5)
    knn_hist.fit(X_histopath_scaled, y)
    knn_all = KNeighborsClassifier(n_neighbors=5)
    knn_all.fit(X_all_scaled, y)

    y_pred_clin = knn_clin.predict(X_clinical_scaled)
    y_pred_hist = knn_hist.predict(X_histopath_scaled)
    y_pred_all = knn_all.predict(X_all_scaled)
    
    acc_clin = accuracy_score(y, y_pred_clin)
    acc_hist = accuracy_score(y, y_pred_hist)
    acc_all = accuracy_score(y, y_pred_all)
    
    print(f"Training Accuracy (Clinical): {acc_clin:.4f}")
    print(f"Training Accuracy (Histopathological): {acc_hist:.4f}")
    print(f"Training Accuracy (All features): {acc_all:.4f}")

    clinical_features = symptoms_to_features(clinical_symptoms_text, clinical_cols)
    histopath_features = symptoms_to_features(histopath_symptoms_text, histopath_cols)
    
    scaler_clin = StandardScaler()
    scaler_hist = StandardScaler()
    scaler_all = StandardScaler()
    
    scaler_clin.fit(X_clinical)
    scaler_hist.fit(X_histopath)
    scaler_all.fit(X_all)
    
    clinical_features_scaled = scaler_clin.transform(clinical_features)
    histopath_features_scaled = scaler_hist.transform(histopath_features)
    
    clinical_features_no_age = clinical_features[0, :-1]
    combined_features = np.concatenate([clinical_features_no_age, histopath_features[0], clinical_features[0, -1:]])
    combined_features = combined_features.reshape(1, -1)
    combined_features_scaled = scaler_all.transform(combined_features)
    
    pred_clin = knn_clin.predict(clinical_features_scaled)[0]
    pred_hist = knn_hist.predict(histopath_features_scaled)[0]
    pred_combined = knn_all.predict(combined_features_scaled)[0]

    # disease_map to get the disease name:
    disease_clin_name = disease_map[pred_clin]
    disease_hist_name = disease_map[pred_hist]
    disease_combined_name = disease_map[pred_combined]

    print(f"Predicted disease (clinical symptoms): {disease_clin_name}")
    print(f"Predicted disease (histopathological symptoms): {disease_hist_name}")
    print(f"Predicted disease (all symptoms combined): {disease_combined_name}")

    # loss function:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(k_neighbors_list, loss_clin, 'b-o', label='Clinical Features', linewidth=2, markersize=6)
    plt.plot(k_neighbors_list, loss_hist, 'r-s', label='Histopathological Features', linewidth=2, markersize=6)
    plt.plot(k_neighbors_list, loss_all, 'g-^', label='All Features', linewidth=2, markersize=6)
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Misclassification Rate (Loss)')
    plt.title('KNN Loss Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(max(loss_clin), max(loss_hist), max(loss_all)) * 1.1)

    # confusion matrices:
    plt.subplot(1, 2, 2)
    cm_all = confusion_matrix(y, y_pred_all)
    sns.heatmap(cm_all, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - All Features')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    cm_clin = confusion_matrix(y, y_pred_clin)
    sns.heatmap(cm_clin, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix - Clinical Features')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    cm_hist = confusion_matrix(y, y_pred_hist)
    sns.heatmap(cm_hist, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('Confusion Matrix - Histopathological Features')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    sns.heatmap(cm_all, annot=True, fmt='d', cmap='Blues', ax=axes[2])
    axes[2].set_title('Confusion Matrix - All Features')
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()

    return disease_clin_name, disease_hist_name, disease_combined_name

# use two different clustering algorithms and see how well these attributes can determine the disease type (model4 and model5).
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
    X_age, X_all, X_clinical, X_histopath, X_clinical_scaled, X_histopath_scaled, X_all_scaled, y, le = load_and_preprocess()

    # Example usage of the new model2_random_forest function with symptom text arrays
    example_clinical = ["Erythema", "Scaling", "Itching", "Age:45"]
    example_histopath = ["Melanin", "Acanthosis", "Hyperkeratosis"]
    
    print("=== RANDOM FOREST MODEL ===")
    model2_random_forest(example_clinical, example_histopath, y, le)
    
    print("=== KNN MODEL ===")
    model3_knn(example_clinical, example_histopath, y, le)

if __name__ == "__main__":  # If this script is run directly
    main()  # Call the main function
    