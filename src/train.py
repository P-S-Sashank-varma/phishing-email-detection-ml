import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

def train_model(data_file, model_file):
    # Load preprocessed data
    with open(data_file, "rb") as f:
        X, y, _ = pickle.load(f)

    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y  # Maintain class distribution
    )

    # Train a Random Forest model with optimized parameters
    model = RandomForestClassifier(
        n_estimators=200,      # Number of trees
        max_depth=15,          # Maximum depth of trees
        min_samples_split=5,   # Minimum samples to split a node
        min_samples_leaf=2,    # Minimum samples in leaf node
        random_state=42,
        n_jobs=-1,             # Use all processors
        class_weight='balanced'  # Handle class imbalance
    )
    
    print("\nTraining Random Forest Classifier...")
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluate the model
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)

    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Testing Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1-Score:          {f1:.4f}")
    print("="*50)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    print("\nConfusion Matrix:")
    print(f"True Negatives:  {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives:  {cm[1][1]}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['Legitimate', 'Phishing']))

    # Feature Importance
    feature_importance = model.feature_importances_
    print(f"\nTop 10 Most Important Features:")
    top_features_idx = np.argsort(feature_importance)[-10:][::-1]
    for idx, fi in enumerate(feature_importance[top_features_idx], 1):
        print(f"  {idx}. Importance: {fi:.4f}")

    # Save the trained model
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    print(f"\n✓ Model saved to {model_file}")
    
    return test_accuracy

if __name__ == "__main__":
    accuracy = train_model("data/preprocessed_data.pkl", "models/phishing_detector.pkl")
    if accuracy >= 0.90:
        print(f"\n✓ SUCCESS: Achieved {accuracy*100:.2f}% accuracy (>= 90%)")
    else:
        print(f"\n✗ Target accuracy not met. Current: {accuracy*100:.2f}%")
