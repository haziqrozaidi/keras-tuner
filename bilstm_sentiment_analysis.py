import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
import matplotlib.pyplot as plt
import pickle
import os

# Create output directory for BiLSTM tuned results
output_dir = os.path.join('results', 'bilstm_tuned')
os.makedirs(output_dir, exist_ok=True)
print(f"Saving results to {output_dir}")

# Download necessary NLTK packages
nltk.download('stopwords')
nltk.download('vader_lexicon')  # Download VADER lexicon for sentiment analysis

def get_sentiment_label(text):
    if isinstance(text, float) and np.isnan(text):
        return 'Neutral'  # Handle NaN values
    
    scores = sid.polarity_scores(str(text))
    compound_score = scores['compound']
    
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def clean_text(text):
    # Handle NaN values
    if isinstance(text, float) and np.isnan(text):
        return ""
    
    # Remove URLs, mentions, hashtags, and special characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text), flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Convert to lowercase and tokenize
    words = text.lower().split()
    
    # Remove stopwords and stem
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

# Define model builder function for Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_len))
    
    # Tune LSTM Units
    lstm_units = hp.Int('lstm_units', min_value=64, max_value=256, step=32)
    
    # Tune LSTM Dropout and Recurrent Dropout
    lstm_dropout = hp.Float('lstm_dropout', min_value=0.0, max_value=0.5, step=0.1)
    lstm_recurrent_dropout = hp.Float('lstm_recurrent_dropout', min_value=0.0, max_value=0.5, step=0.1)
    
    model.add(Bidirectional(LSTM(units=lstm_units, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout)))
    
    # Tune Dense Units
    dense_units = hp.Int('dense_units', min_value=32, max_value=128, step=32)
    
    # Tune Dense Dropout
    dense_dropout = hp.Float('dense_dropout', min_value=0.0, max_value=0.5, step=0.1)
    
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dense_dropout))
    
    # Adjust the output layer based on number of classes
    if num_classes == 2:  # Binary
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:  # Multi-class
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'
    
    # Tune Learning Rate
    learning_rate = hp.Float('learning_rate', min_value=0.0001, max_value=0.01, sampling='log')
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['accuracy']
    )
    return model

def predict_sentiment(text, model, tokenizer, sentiment_mapping, max_len, num_classes):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=max_len)
    
    if num_classes == 2:  # Binary classification
        prediction = model.predict(padded)[0][0]
        sentiment_idx = 1 if prediction >= 0.5 else 0
        confidence = prediction if prediction >= 0.5 else 1 - prediction
    else:  # Multi-class classification
        prediction = model.predict(padded)[0]
        sentiment_idx = np.argmax(prediction)
        confidence = prediction[sentiment_idx]
    
    # Convert index back to sentiment label
    for label, idx in sentiment_mapping.items():
        if idx == sentiment_idx:
            sentiment = label
            break
    
    return sentiment, float(confidence)

def plot_model_metrics(accuracy, precision, recall, f1, output_path):
    """
    Creates a bar chart showing the model's performance metrics.
    
    Args:
        accuracy: Model accuracy value
        precision: Model precision score
        recall: Model recall score
        f1: Model F1-score
        output_path: Path to save the chart
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]
    
    # Convert to percentage for better visualization
    values_percent = [val * 100 for val in values]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values_percent, color=['#2C7BB6', '#D7191C', '#FDAE61', '#1A9641'])
    
    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.title('BiLSTM Model Performance Metrics with Tuned Hyperparameters', fontsize=16)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.ylim(0, 105)  # Set y-axis limit to accommodate the labels
    
    # Add grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Performance metrics visualization saved to {output_path}")

def plot_hyperparameter_comparison(best_hyperparameters, output_path):
    """
    Creates a visualization of the tuned hyperparameters.
    
    Args:
        best_hyperparameters: Dictionary containing the best hyperparameters
        output_path: Path to save the chart
    """
    # Prepare data for plotting
    params = list(best_hyperparameters.keys())
    values = list(best_hyperparameters.values())
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create separate plots for different types of hyperparameters
    # Plot for integer hyperparameters (units)
    int_params = ['lstm_units', 'dense_units']
    int_values = [best_hyperparameters[p] for p in int_params if p in best_hyperparameters]
    if int_values:
        plt.subplot(2, 2, 1)
        plt.bar(int_params, int_values, color='skyblue')
        plt.title('Architecture Parameters')
        plt.ylabel('Units')
        for i, v in enumerate(int_values):
            plt.text(i, v + 5, str(v), ha='center')
    
    # Plot for float hyperparameters (dropout rates)
    dropout_params = ['lstm_dropout', 'lstm_recurrent_dropout', 'dense_dropout']
    dropout_values = [best_hyperparameters[p] for p in dropout_params if p in best_hyperparameters]
    if dropout_values:
        plt.subplot(2, 2, 2)
        plt.bar(dropout_params, dropout_values, color='lightgreen')
        plt.title('Regularization Parameters')
        plt.ylabel('Dropout Rate')
        for i, v in enumerate(dropout_values):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
    
    # Plot for learning rate (separate because of scale difference)
    if 'learning_rate' in best_hyperparameters:
        plt.subplot(2, 2, 3)
        plt.bar(['learning_rate'], [best_hyperparameters['learning_rate']], color='salmon')
        plt.title('Optimization Parameters')
        plt.ylabel('Learning Rate')
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        plt.text(0, best_hyperparameters['learning_rate'] + 0.0005, 
                 f"{best_hyperparameters['learning_rate']:.6f}", ha='center')
    
    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.text(0.1, 0.5, f"Best Trial Validation Accuracy: {best_val_accuracy:.4f}", fontsize=12)
    plt.text(0.1, 0.4, f"Best Trial Validation Loss: {best_val_loss:.4f}", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Hyperparameter visualization saved to {output_path}")

if __name__ == "__main__":
    # Log execution time
    start_time = pd.Timestamp.now()
    print(f"BiLSTM Analysis with Keras Tuner started at: {start_time}")
    
    # 1. Load dataset
    df = pd.read_csv('tweets_dataset.csv')
    print("Original columns:", df.columns.tolist())
    
    # 2. Use VADER for initial sentiment labeling
    sid = SentimentIntensityAnalyzer()
    df['Sentiment'] = df['Content'].apply(get_sentiment_label)
    print("Sentiment distribution from VADER:", df['Sentiment'].value_counts())
    
    # 3. Text Preprocessing
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    df['Cleaned_Content'] = df['Content'].apply(clean_text)
    
    # 4. Encode sentiment labels
    encoder = LabelEncoder()
    df['Encoded_Sentiment'] = encoder.fit_transform(df['Sentiment'])
    
    # Save original sentiment labels 
    sentiment_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print("Sentiment mapping:", sentiment_mapping)
    
    # 5. Split data
    X = df['Cleaned_Content'].values
    y = df['Encoded_Sentiment'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 6. Tokenize and pad sequences
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Find the maximum sequence length
    max_len = max(len(x) for x in X_train_seq)
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
    
    # Check for class imbalance
    print("Class distribution in training data:", np.bincount(y_train))
    
    # Determine number of classes
    num_classes = len(np.unique(y_train))
    print(f"Number of classes: {num_classes}")
    
    # 7. Setup Keras Tuner
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=15,
        factor=3,
        directory=os.path.join(output_dir, 'tuner_results'),
        project_name='sentiment_bilstm_tuning'
    )
    
    # Print search space summary
    print(tuner.search_space_summary())
    
    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Start the hyperparameter search
    print("\nStarting hyperparameter search...")
    tuner.search(
        X_train_pad, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping]
    )
    
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\nOptimal Hyperparameters Found:")
    print(f"LSTM Units: {best_hps.get('lstm_units')}")
    print(f"LSTM Dropout: {best_hps.get('lstm_dropout')}")
    print(f"LSTM Recurrent Dropout: {best_hps.get('lstm_recurrent_dropout')}")
    print(f"Dense Units: {best_hps.get('dense_units')}")
    print(f"Dense Dropout: {best_hps.get('dense_dropout')}")
    print(f"Learning Rate: {best_hps.get('learning_rate')}")
    
    # Save the best hyperparameters to a file
    best_hyperparameters = {
        'lstm_units': best_hps.get('lstm_units'),
        'lstm_dropout': best_hps.get('lstm_dropout'),
        'lstm_recurrent_dropout': best_hps.get('lstm_recurrent_dropout'),
        'dense_units': best_hps.get('dense_units'),
        'dense_dropout': best_hps.get('dense_dropout'),
        'learning_rate': best_hps.get('learning_rate')
    }
    
    with open(os.path.join(output_dir, 'best_hyperparameters.txt'), 'w') as f:
        for param, value in best_hyperparameters.items():
            f.write(f"{param}: {value}\n")
    
    # Build the model with the optimal hyperparameters
    model = tuner.hypermodel.build(best_hps)
    
    # Get the best trial information
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    best_val_accuracy = best_trial.metrics.get_best_value('val_accuracy')
    best_val_loss = best_trial.metrics.get_best_value('val_loss')
    
    # Train the model with the optimal hyperparameters
    print("\nTraining final model with optimal hyperparameters...")
    history = model.fit(
        X_train_pad, y_train,
        epochs=15,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping]
    )
    
    # 8. Evaluate model
    tuned_loss, tuned_accuracy = model.evaluate(X_test_pad, y_test)
    print(f"Tuned BiLSTM Model - Loss: {tuned_loss}, Accuracy: {tuned_accuracy}")
    
    print("\n--- Detailed Performance Metrics ---")
    
    # Calculate predictions
    y_pred_tuned = model.predict(X_test_pad)
    if num_classes > 2:  # Multi-class case
        y_pred_tuned = np.argmax(y_pred_tuned, axis=1)
    else:  # Binary case
        y_pred_tuned = (y_pred_tuned > 0.5).astype(int).flatten()
    
    # Calculate and display BiLSTM metrics
    print("\nTuned BiLSTM Model Detailed Metrics:")
    classification_report_str = classification_report(y_test, y_pred_tuned)
    print(classification_report_str)
    tuned_precision = precision_score(y_test, y_pred_tuned, average='weighted')
    tuned_recall = recall_score(y_test, y_pred_tuned, average='weighted')
    tuned_f1 = f1_score(y_test, y_pred_tuned, average='weighted')
    
    # Format metrics as percentages
    print(f"Tuned BiLSTM Metrics Summary:")
    print(f"Accuracy: {tuned_accuracy:.2%}")
    print(f"Precision: {tuned_precision:.2%}")
    print(f"Recall: {tuned_recall:.2%}")
    print(f"F1-Score: {tuned_f1:.2%}")
    
    # Calculate actual number of epochs trained
    actual_epochs = len(history.history['loss'])
    print(f"\nActual epochs trained - Tuned BiLSTM: {actual_epochs}")
    
    # 9. Plot training history
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Tuned BiLSTM Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Tuned BiLSTM Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_performance.png'))
    
    # Create and save performance metrics bar chart
    plot_model_metrics(
        tuned_accuracy, 
        tuned_precision, 
        tuned_recall, 
        tuned_f1, 
        os.path.join(output_dir, 'performance_metrics_chart.png')
    )
    
    # Create and save hyperparameter comparison chart
    plot_hyperparameter_comparison(
        best_hyperparameters, 
        os.path.join(output_dir, 'hyperparameter_comparison.png')
    )
    
    # 10. Predict sentiment for sample tweets
    sample_tweets = [
        "AI is going to revolutionize the job market in Malaysia!",
        "Worried about losing my job to AI in the coming years",
        "The impact of AI on Malaysian jobs remains to be seen"
    ]
    
    print("\nTuned BiLSTM Model Predictions:")
    sample_predictions = []
    for tweet in sample_tweets:
        sentiment, confidence = predict_sentiment(tweet, model, tokenizer, 
                                               sentiment_mapping, max_len, num_classes)
        print(f"Tweet: {tweet}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})\n")
        sample_predictions.append({
            'Tweet': tweet,
            'Sentiment': sentiment,
            'Confidence': confidence
        })
    
    # Save sample predictions
    pd.DataFrame(sample_predictions).to_csv(
        os.path.join(output_dir, 'sample_predictions.csv'), index=False
    )
    
    # 11. Save the model and supporting files
    model_path = os.path.join(output_dir, 'sentiment_model_tuned.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save tokenizer and encoder
    with open(os.path.join(output_dir, 'tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(output_dir, 'label_encoder.pickle'), 'wb') as handle:
        pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save model architecture as JSON
    model_json = model.to_json()
    with open(os.path.join(output_dir, 'model_architecture.json'), 'w') as json_file:
        json_file.write(model_json)
    
    # Save the sentiment distribution to a CSV file
    sentiment_distribution = df['Sentiment'].value_counts().reset_index()
    sentiment_distribution.columns = ['Sentiment', 'Count']
    sentiment_distribution.to_csv(
        os.path.join(output_dir, 'sentiment_distribution.csv'), index=False
    )
    
    # Save the classification report to a text file
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(classification_report_str)
    
    # Create a performance summary DataFrame and save to CSV
    performance_metrics = pd.DataFrame({
        'Model': ['Tuned BiLSTM'],
        'Accuracy': [tuned_accuracy],
        'Precision': [tuned_precision],
        'Recall': [tuned_recall],
        'F1_Score': [tuned_f1],
        'Epochs_Trained': [actual_epochs]
    })
    performance_metrics.to_csv(
        os.path.join(output_dir, 'performance_metrics.csv'), index=False
    )
    
    # Save history to CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    # Log completion
    end_time = pd.Timestamp.now()
    elapsed_time = end_time - start_time
    print(f"\nTuned BiLSTM Analysis completed at: {end_time}")
    print(f"Total execution time: {elapsed_time}")
    
    # Save execution log
    with open(os.path.join(output_dir, 'execution_log.txt'), 'w') as f:
        f.write(f"Analysis started: {start_time}\n")
        f.write(f"Analysis completed: {end_time}\n")
        f.write(f"Total execution time: {elapsed_time}\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Max sequence length: {max_len}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Testing samples: {len(X_test)}\n")
        f.write(f"Class distribution: {np.bincount(y_train).tolist()}\n")
        f.write(f"Final epochs: {actual_epochs}\n")
        f.write(f"Final accuracy: {tuned_accuracy:.4f}\n")
        f.write(f"Final loss: {tuned_loss:.4f}\n")
        f.write("\nBest Hyperparameters:\n")
        for param, value in best_hyperparameters.items():
            f.write(f"{param}: {value}\n")
        
    print(f"\nAll results saved to {output_dir}")
    