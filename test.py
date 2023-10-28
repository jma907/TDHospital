import pandas as pd

# Load the dataset
data = pd.read_csv('TD_HOSPITAL_TRAIN.csv')

# Display the first few rows of the dataset to understand its structure
data.head()


 

missing_values = data.isnull().sum()
print(data.isna().any(axis=1).sum())

# Calculate the percentage of missing values for each column
missing_percentage = (missing_values / len(data)) * 100

print(missing_percentage.sort_values(ascending=False))


print(missing_values)


# Drop the 'pdeath' column
data.drop(columns=['pdeath'], inplace=True)

# Standardize the 'sex' column
data['sex'] = data['sex'].str.lower().replace({"m": "male", "f": "female"})

# Display the unique values in the 'sex' column to confirm standardization
data['sex'].unique()


print(array(['male', 'female', '1'], dtype=object))


# Drop columns with more than 50% missing values
columns_to_drop = missing_percentage[missing_percentage > 50].index
data_cleaned = data.drop(columns=columns_to_drop)

# Check how many rows have any NA values
rows_with_na = data_cleaned.isna().any(axis=1).sum()

# Calculate the percentage of rows with NA values relative to the entire dataset
percentage_rows_with_na = (rows_with_na / len(data_cleaned)) * 100

rows_with_na, percentage_rows_with_na


def data_preprocessing(df):
    '''    
    Preprocesses the data by:
    - Keeping specified columns
    - Replacing empty strings and NaN values with 0
    - One-hot encoding categorical columns
    '''
    
    col_to_keep = [
        'cost', 'bloodchem1', 'income', 'psych2', 'bloodchem3', 'bloodchem6',
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
    
    # Apply one-hot encoding
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_cols = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Merge one-hot encoded columns with the original dataframe and drop original categorical columns
    df = pd.concat([df, encoded_df], axis=1).drop(columns=categorical_cols)
    
    return df

# Test the function on the data
preprocessed_data = data_preprocessing(data_cleaned)
preprocessed_data.head()