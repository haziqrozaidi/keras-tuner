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
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pickle
import os

# Create output directory for LSTM results
output_dir = os.path.join('results', 'lstm')
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

def create_lstm_model(num_classes=1, max_len=100):
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_len))
    model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    
    # Adjust the output layer based on number of classes
    if num_classes == 2:  # Binary (e.g., Positive vs Negative)
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:  # Multi-class (e.g., Positive, Negative, Neutral)
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'
    
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
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
    
    plt.title('LSTM Model Performance Metrics with Current Hyperparameters', fontsize=16)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.ylim(0, 105)  # Set y-axis limit to accommodate the labels
    
    # Add grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Performance metrics visualization saved to {output_path}")

if __name__ == "__main__":
    # Log execution time
    start_time = pd.Timestamp.now()
    print(f"LSTM Analysis started at: {start_time}")
    
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
    
    # 7. Train LSTM model
    lstm_model = create_lstm_model(num_classes, max_len)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    lstm_history = lstm_model.fit(
        X_train_pad, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping]
    )
    
    # 8. Evaluate model
    lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test_pad, y_test)
    print(f"LSTM Model - Loss: {lstm_loss}, Accuracy: {lstm_accuracy}")
    
    print("\n--- Detailed Performance Metrics ---")
    
    # Calculate predictions
    y_pred_lstm = lstm_model.predict(X_test_pad)
    if num_classes > 2:  # Multi-class case
        y_pred_lstm = np.argmax(y_pred_lstm, axis=1)
    else:  # Binary case
        y_pred_lstm = (y_pred_lstm > 0.5).astype(int).flatten()
    
    # Calculate and display LSTM metrics
    print("\nLSTM Model Detailed Metrics:")
    classification_report_str = classification_report(y_test, y_pred_lstm)
    print(classification_report_str)
    lstm_precision = precision_score(y_test, y_pred_lstm, average='weighted')
    lstm_recall = recall_score(y_test, y_pred_lstm, average='weighted')
    lstm_f1 = f1_score(y_test, y_pred_lstm, average='weighted')
    
    # Format metrics as percentages
    print(f"LSTM Metrics Summary:")
    print(f"Accuracy: {lstm_accuracy:.2%}")
    print(f"Precision: {lstm_precision:.2%}")
    print(f"Recall: {lstm_recall:.2%}")
    print(f"F1-Score: {lstm_f1:.2%}")
    
    # Calculate actual number of epochs trained
    actual_lstm_epochs = len(lstm_history.history['loss'])
    print(f"\nActual epochs trained - LSTM: {actual_lstm_epochs}")
    
    # 9. Plot training history
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(lstm_history.history['accuracy'], label='Train')
    plt.plot(lstm_history.history['val_accuracy'], label='Validation')
    plt.title('LSTM Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(lstm_history.history['loss'], label='Train')
    plt.plot(lstm_history.history['val_loss'], label='Validation')
    plt.title('LSTM Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_performance.png'))
    
    # NEW: Create and save a performance metrics bar chart
    plot_model_metrics(
        lstm_accuracy, 
        lstm_precision, 
        lstm_recall, 
        lstm_f1, 
        os.path.join(output_dir, 'performance_metrics_chart.png')
    )
    
    # 10. Predict sentiment for sample tweets
    sample_tweets = [
        "AI is going to revolutionize the job market in Malaysia!",
        "Worried about losing my job to AI in the coming years",
        "The impact of AI on Malaysian jobs remains to be seen"
    ]
    
    print("\nLSTM Model Predictions:")
    sample_predictions = []
    for tweet in sample_tweets:
        sentiment, confidence = predict_sentiment(tweet, lstm_model, tokenizer, 
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
    model_path = os.path.join(output_dir, 'sentiment_model.h5')
    lstm_model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save tokenizer and encoder
    with open(os.path.join(output_dir, 'tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(output_dir, 'label_encoder.pickle'), 'wb') as handle:
        pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save model architecture as JSON
    model_json = lstm_model.to_json()
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
        'Model': ['LSTM'],
        'Accuracy': [lstm_accuracy],
        'Precision': [lstm_precision],
        'Recall': [lstm_recall],
        'F1_Score': [lstm_f1],
        'Epochs_Trained': [actual_lstm_epochs]
    })
    performance_metrics.to_csv(
        os.path.join(output_dir, 'performance_metrics.csv'), index=False
    )
    
    # Save history to CSV
    history_df = pd.DataFrame(lstm_history.history)
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    # Log completion
    end_time = pd.Timestamp.now()
    elapsed_time = end_time - start_time
    print(f"\nLSTM Analysis completed at: {end_time}")
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
        f.write(f"Final epochs: {actual_lstm_epochs}\n")
        f.write(f"Final accuracy: {lstm_accuracy:.4f}\n")
        f.write(f"Final loss: {lstm_loss:.4f}\n")
        
    print(f"\nAll results saved to {output_dir}")
    