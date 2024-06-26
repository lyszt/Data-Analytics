import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv('logs.csv',encoding='latin-1')
df = df.fillna(0)

# Preprocessing
df['Time_to_finish'] = (df['Time_to_finish']).fillna(0)
df['Pages'] = (df['Pages']).fillna(0)
df['Pages_Read'] = (df['Pages_Read']).fillna(0)

df['Year'] = df['Year'].astype(int)

df['Author'] = df['Author'].astype(str)
df['Author_Nationality'] = df['Author_Nationality'].astype(str)
df['Genre'] = df['Genre'].astype(str)
df['Language'] = df['Language'].astype(str)
df['Source'] = df['Source'].astype(str)
df['Status'] = df['Status'].astype(str)
df['Key_Emotion'] = df['Key_Emotion'].astype(str)
df['Secondary_E'] = df['Secondary_E'].astype(str)
df['Topic'] = df['Topic'].astype(str)



# Encode categorical variables
label_encoder = LabelEncoder()

df['Author_encoded'] = label_encoder.fit_transform(df['Author'])
df['Author_nationality_encoded'] = label_encoder.fit_transform(df['Author_Nationality'])
df['Genre_encoded'] = label_encoder.fit_transform(df['Genre'])
df['Language_encoded'] = label_encoder.fit_transform(df['Language'])
df['Topic_encoded'] = label_encoder.fit_transform(df['Topic'])
df['Source_encoded'] = label_encoder.fit_transform(df['Source'])
df['Status_encoded'] = label_encoder.fit_transform(df['Status'])
df['Year_encoded'] = label_encoder.fit_transform(df['Year'])
df['Keyemotion_encoded'] = label_encoder.fit_transform(df['Key_Emotion'])
df['Secemotion_encoded'] = label_encoder.fit_transform(df['Secondary_E'])

# Features and target variable
features = df[['Author_encoded', 'Author_nationality_encoded', 'Genre_encoded', 'Language_encoded', 'Topic_encoded',
               'Source_encoded', 'Status_encoded', 'Keyemotion_encoded', 'Secemotion_encoded', 'Time_to_finish','Year_encoded']]
target = df['Liked']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# Function to predict if you'll like a new book
def predict_liking(book_details):
    book_df = pd.DataFrame([book_details])
    df['Author_encoded'] = label_encoder.fit_transform(df['Author'])
    df['Author_nationality_encoded'] = label_encoder.fit_transform(df['Author_Nationality'])
    df['Genre_encoded'] = label_encoder.fit_transform(df['Genre'])
    df['Language_encoded'] = label_encoder.fit_transform(df['Language'])
    df['Topic_encoded'] = label_encoder.fit_transform(df['Topic'])
    df['Year_encoded'] = label_encoder.fit_transform(df['Year'])
    df['Source_encoded'] = label_encoder.fit_transform(df['Source'])
    df['Status_encoded'] = label_encoder.fit_transform(df['Status'])
    df['Keyemotion_encoded'] = label_encoder.fit_transform(df['Key_Emotion'])
    df['Secemotion_encoded'] = label_encoder.fit_transform(df['Secondary_E'])
    return model.predict(book_df)
