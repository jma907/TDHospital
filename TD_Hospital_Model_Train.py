import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder


def data_preprocessing(df):
    '''    
    Preprocesses the data by:
    - Keeping specified columns
    - Replacing empty strings and NaN values with 0
    - One-hot encoding categorical columns
    '''
    
    col_to_keep = [
        'bloodchem3', 'bloodchem6',
        'sleep', 'bloodchem5', 'education', 'disability', 'totalcost', 'confidence',
        'blood', 'administratorcost', 'psych5', 'bloodchem2', 'race', 'dnr', 
        'information', 'timeknown', 'diabetes', 'psych6', 'cancer', 'extraprimary',
        'age', 'primary', 'pain', 'meals', 'breathing', 'comorbidity', 'bp',
        'psych3', 'dose', 'psych1', 'heart', 'temperature', 'sex', 'reflex', 'death'
    ]
    
    df = df[col_to_keep]
    df.replace('', 0, inplace=True)
    df.fillna(0, inplace=True)
    
    # Identify categorical columns that need one-hot encoding
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Convert all categorical columns to string type
    df[categorical_cols] = df[categorical_cols].astype(str)
    
    # Apply one-hot encoding
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_cols = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Merge one-hot encoded columns with the original dataframe and drop original categorical columns
    df = pd.concat([df, encoded_df], axis=1).drop(columns=categorical_cols)
    
    return df

# Test the function on the data

def split_feature_label(df):
    y = df['death']
    X = df.drop(columns=['death'])
    return y, X
    # print(X)
    # print(y)

    # death_0 = y.tolist().count(0)
    # death_1 = y.tolist().count(1)
    # percent_death_0 = 100 * death_0 / (death_0 + death_1)
    # percent_death_1 = 100 * death_1 / (death_0 + death_1)
    # print(f'Survived: {death_0}, or {percent_death_0:.2f}%')
    # print(f'Died: {death_1}, or {percent_death_1:.2f}%')

def standardize(X):
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(X.select_dtypes(include=['float64']))
    X[X.select_dtypes(include=['float64']).columns] = X_numeric
    return X

def train_model(X, y):
    # Split data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=.3, random_state=42)

    # Define the neural network model
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),  # Input layer
        layers.Dense(128, activation='relu'),     # Hidden layer with 128 neurons and ReLU activation
        layers.Dense(64, activation='relu'),      # Another hidden layer with 64 neurons and ReLU activation
        layers.Dense(1, activation='sigmoid')     # Output layer with sigmoid activation for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    model.save('example.h5')
    
    print(f'Test accuracy: {test_accuracy}')

    # Optionally, you can plot training history to visualize model performance
    import matplotlib.pyplot as plt

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()



if __name__ == "__main__":
    data_path = './TD_HOSPITAL_TRAIN.csv'
    df = pd.read_csv(data_path)
    cleaned_data = data_preprocessing(df)
    y, X = split_feature_label(cleaned_data)
    X = standardize(X)
    train_model(X, y)
    