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
import tensorflow as tf
import keras_tuner as kt
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

# Model building function for Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(Embedding(
        input_dim=5000,
        output_dim=128,
        input_length=max_len
    ))
    
    # Tune LSTM units
    lstm_units = hp.Int('lstm_units', min_value=32, max_value=256, step=32)
    
    # Tune dropout rate for LSTM
    lstm_dropout = hp.Float('lstm_dropout', min_value=0.1, max_value=0.5, step=0.1)
    lstm_recurrent_dropout = hp.Float('lstm_recurrent_dropout', min_value=0.1, max_value=0.5, step=0.1)
    
    model.add(LSTM(units=lstm_units, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout))
    
    # Tune dense layer units
    dense_units = hp.Int('dense_units', min_value=32, max_value=128, step=32)
    model.add(Dense(dense_units, activation='relu'))
    
    # Tune dropout rate for dense layer
    dense_dropout = hp.Float('dense_dropout', min_value=0.1, max_value=0.7, step=0.1)
    model.add(Dropout(dense_dropout))
    
    # Adjust the output layer based on number of classes
    if num_classes == 2:  # Binary (e.g., Positive vs Negative)
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:  # Multi-class (e.g., Positive, Negative, Neutral)
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'
    
    # Tune learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['accuracy']
    )
    return model

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

if __name__ == "__main__":
    # Log execution time
    start_time = pd.Timestamp.now()
    print(f"LSTM Analysis with Keras Tuner started at: {start_time}")
    
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
    
    # Create hyperparameter tuner directory
    tuner_dir = os.path.join(output_dir, 'keras_tuner')
    os.makedirs(tuner_dir, exist_ok=True)
    
    # 7. Setup Keras Tuner
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=15,
        factor=3,
        directory=tuner_dir,
        project_name='sentiment_analysis'
    )
    
    # Define early stopping callback for the search
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    
    # Print search space summary
    tuner.search_space_summary()
    
    # Perform hyperparameter tuning
    print("\nStarting hyperparameter search...")
    tuner.search(
        X_train_pad, y_train,
        validation_split=0.2,
        epochs=10,
        callbacks=[stop_early],
        verbose=1
    )
    
    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Print best hyperparameters
    print("\nBest Hyperparameters:")
    print(f"LSTM Units: {best_hps.get('lstm_units')}")
    print(f"LSTM Dropout: {best_hps.get('lstm_dropout')}")
    print(f"LSTM Recurrent Dropout: {best_hps.get('lstm_recurrent_dropout')}")
    print(f"Dense Units: {best_hps.get('dense_units')}")
    print(f"Dense Dropout: {best_hps.get('dense_dropout')}")
    print(f"Learning Rate: {best_hps.get('learning_rate')}")
    
    # Save best hyperparameters to file
    with open(os.path.join(tuner_dir, 'best_hyperparameters.txt'), 'w') as f:
        f.write(f"LSTM Units: {best_hps.get('lstm_units')}\n")
        f.write(f"LSTM Dropout: {best_hps.get('lstm_dropout')}\n")
        f.write(f"LSTM Recurrent Dropout: {best_hps.get('lstm_recurrent_dropout')}\n")
        f.write(f"Dense Units: {best_hps.get('dense_units')}\n")
        f.write(f"Dense Dropout: {best_hps.get('dense_dropout')}\n")
        f.write(f"Learning Rate: {best_hps.get('learning_rate')}\n")
        
    # Build the model with the best hyperparameters and train it
    best_model = tuner.hypermodel.build(best_hps)
    
    # Define batch size - could also be tuned but we'll set it for simplicity
    batch_size = 32
    
    # Determine the number of epochs for final training
    epochs = 15
    
    print(f"\nTraining final model with best hyperparameters (batch size: {batch_size}, epochs: {epochs})...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = best_model.fit(
        X_train_pad, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 8. Evaluate model
    lstm_loss, lstm_accuracy = best_model.evaluate(X_test_pad, y_test)
    print(f"Tuned LSTM Model - Loss: {lstm_loss}, Accuracy: {lstm_accuracy}")
    
    print("\n--- Detailed Performance Metrics ---")
    
    # Calculate predictions
    y_pred_lstm = best_model.predict(X_test_pad)
    if num_classes > 2:  # Multi-class case
        y_pred_lstm = np.argmax(y_pred_lstm, axis=1)
    else:  # Binary case
        y_pred_lstm = (y_pred_lstm > 0.5).astype(int).flatten()
    
    # Calculate and display LSTM metrics
    print("\nTuned LSTM Model Detailed Metrics:")
    classification_report_str = classification_report(y_test, y_pred_lstm)
    print(classification_report_str)
    lstm_precision = precision_score(y_test, y_pred_lstm, average='weighted')
    lstm_recall = recall_score(y_test, y_pred_lstm, average='weighted')
    lstm_f1 = f1_score(y_test, y_pred_lstm, average='weighted')
    
    # Format metrics as percentages
    print(f"Tuned LSTM Metrics Summary:")
    print(f"Accuracy: {lstm_accuracy:.2%}")
    print(f"Precision: {lstm_precision:.2%}")
    print(f"Recall: {lstm_recall:.2%}")
    print(f"F1-Score: {lstm_f1:.2%}")
    
    # Calculate actual number of epochs trained
    actual_lstm_epochs = len(history.history['loss'])
    print(f"\nActual epochs trained - LSTM: {actual_lstm_epochs}")
    
    # 9. Plot training history
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Tuned LSTM Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Tuned LSTM Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tuned_model_performance.png'))
    
    # 10. Predict sentiment for sample tweets
    sample_tweets = [
        "AI is going to revolutionize the job market in Malaysia!",
        "Worried about losing my job to AI in the coming years",
        "The impact of AI on Malaysian jobs remains to be seen"
    ]
    
    print("\nTuned LSTM Model Predictions:")
    sample_predictions = []
    for tweet in sample_tweets:
        sentiment, confidence = predict_sentiment(tweet, best_model, tokenizer, 
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
        os.path.join(output_dir, 'tuned_sample_predictions.csv'), index=False
    )
    
    # 11. Save the model and supporting files
    model_path = os.path.join(output_dir, 'tuned_sentiment_model.h5')
    best_model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save tokenizer and encoder
    with open(os.path.join(output_dir, 'tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(output_dir, 'label_encoder.pickle'), 'wb') as handle:
        pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save model architecture as JSON
    model_json = best_model.to_json()
    with open(os.path.join(output_dir, 'tuned_model_architecture.json'), 'w') as json_file:
        json_file.write(model_json)
    
    # Save the sentiment distribution to a CSV file
    sentiment_distribution = df['Sentiment'].value_counts().reset_index()
    sentiment_distribution.columns = ['Sentiment', 'Count']
    sentiment_distribution.to_csv(
        os.path.join(output_dir, 'sentiment_distribution.csv'), index=False
    )
    
    # Save the classification report to a text file
    with open(os.path.join(output_dir, 'tuned_classification_report.txt'), 'w') as f:
        f.write(classification_report_str)
    
    # Create a performance summary DataFrame and save to CSV
    performance_metrics = pd.DataFrame({
        'Model': ['Tuned LSTM'],
        'Accuracy': [lstm_accuracy],
        'Precision': [lstm_precision],
        'Recall': [lstm_recall],
        'F1_Score': [lstm_f1],
        'Epochs_Trained': [actual_lstm_epochs],
        'Batch_Size': [batch_size],
        'LSTM_Units': [best_hps.get('lstm_units')],
        'LSTM_Dropout': [best_hps.get('lstm_dropout')],
        'LSTM_Recurrent_Dropout': [best_hps.get('lstm_recurrent_dropout')],
        'Dense_Units': [best_hps.get('dense_units')],
        'Dense_Dropout': [best_hps.get('dense_dropout')],
        'Learning_Rate': [best_hps.get('learning_rate')]
    })
    performance_metrics.to_csv(
        os.path.join(output_dir, 'tuned_performance_metrics.csv'), index=False
    )
    
    # Save history to CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(output_dir, 'tuned_training_history.csv'), index=False)
    
    # Log completion
    end_time = pd.Timestamp.now()
    elapsed_time = end_time - start_time
    print(f"\nLSTM Analysis with Keras Tuner completed at: {end_time}")
    print(f"Total execution time: {elapsed_time}")
    
    # Save execution log
    with open(os.path.join(output_dir, 'tuned_execution_log.txt'), 'w') as f:
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
        f.write("\nBest Hyperparameters:\n")
        f.write(f"LSTM Units: {best_hps.get('lstm_units')}\n")
        f.write(f"LSTM Dropout: {best_hps.get('lstm_dropout')}\n")
        f.write(f"LSTM Recurrent Dropout: {best_hps.get('lstm_recurrent_dropout')}\n")
        f.write(f"Dense Units: {best_hps.get('dense_units')}\n")
        f.write(f"Dense Dropout: {best_hps.get('dense_dropout')}\n")
        f.write(f"Learning Rate: {best_hps.get('learning_rate')}\n")
        f.write(f"Batch Size: {batch_size}\n")
    
    print(f"\nAll results saved to {output_dir}")
    