import datetime
import json

import openai
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle


class Predictor:
    def __init__(self):
        self.model = None
        self.train()
        label_encoders = self.load_label_encoders()
        book_details = self.input_book_details()
        prediction = self.predict_liking(book_details, label_encoders)
        if prediction[0] == 1:
            print("You'll like this book!")
        else:
            print("You might not like this book.")

    def train(self):
        df = pd.read_csv('logs.csv', encoding='latin-1')
        df = df.fillna(method='ffill').fillna(method='bfill')  # Improved NaN handling

        # Create separate LabelEncoders for each categorical feature
        label_encoders = {col: LabelEncoder() for col in
                          ['Author_Nationality', 'Genre', 'Language', 'Topic', 'Source', 'Statu s', 'Key_Emotion',
                           'Secondary_E']}

        for col, encoder in label_encoders.items():
            df[f'{col}_encoded'] = encoder.fit_transform(df[col].astype(str))

        with open('label_encoders.pkl', 'wb') as le_file:
            pickle.dump(label_encoders, le_file)

        df['Completed'] = df['Status'].apply(lambda x: 1 if x == 'complete' else 0)

        for col in ['Author_Nationality', 'Genre', 'Language', 'Topic']:
            df[f'{col}_completed_count'] = df.apply(
                lambda row: df[(df[col] == row[col]) & (df['Completed'] == 1)].shape[0], axis=1)

        features = df[['Author_Nationality_encoded', 'Genre_encoded', 'Language_encoded', 'Topic_encoded',
                       'Source_encoded', 'Status_encoded', 'Key_Emotion_encoded', 'Secondary_E_encoded',
                       'Time_to_finish']]
        target = df['Liked']

        # Feature scaling
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

        # Hyperparameter tuning with GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_

        y_pred = self.model.predict(X_test)
        print("Best Parameters:", grid_search.best_params_)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

    def load_label_encoders(self, filename='label_encoders.pkl'):
        with open(filename, 'rb') as le_file:
            label_encoders = pickle.load(le_file)
        return label_encoders

    def predict_liking(self, book_details, label_encoders):
        book_df = pd.DataFrame([book_details])

        for col, encoder in label_encoders.items():
            try:
                book_df[f'{col}_encoded'] = encoder.transform(book_df[col].astype(str))
            except ValueError:
                book_df[f'{col}_encoded'] = -1

        features_input = book_df[[f'{col}_encoded' for col in label_encoders] + ['Time_to_finish']]

        scaler = StandardScaler()
        features_input_scaled = scaler.fit_transform(features_input)

        prediction = self.model.predict(features_input_scaled)
        return prediction

    def input_book_details(self):
        print("Enter book name:")
        book_name = input("")

        with open('token.json','r') as file:
            token = json.load(file)
            api_key = token["token"]
            client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "I'll give you a book name. Based on the book information, complete this python "
                                    "dictionary accordingly with the info and give me it raw, without the variable "
                                    "declaration. Just the dictionary. Don't reply with anything else other than the "
                                    "dictionary. Put it all but the numbers in double quotes. There are some "
                                    "rules.\nPossible inputs:\n\"Author_Nationality\" can ONLY equal European, Brazilian, "
                                    "American/Canadian, Greek and Other\n\"Genre\" can ONLY equal Novel,History,"
                                    "Philosophy,Fantasy,Fable,Article,Science Fiction and Non-Fiction\n\"Topic\" can ONLY "
                                    "equal Self,Brazil,Generic,History,Fantasy,Art,Science,Drama and Society\nCentury "
                                    "is the century in which it mas made. In case it is before christ put "
                                    "-1.\n\nbook_details = { 'Author_Nationality':  \"\", 'Genre': \"\",  "
                                    "'Topic': \"\",'Time_to_finish': Total amount of pages, 'Century':  int}\n\n\n"
                            f"Book name is {book_name}"
                        }
                    ]
                }
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        autocompleted_dict = response.choices[0].message.content
        print(autocompleted_dict)
        autocompleted_dict = json.loads(autocompleted_dict)
        print(autocompleted_dict)
        key_emotion = input("How do you feel about this book?\n (Interesting,Boring,Empathic,Relatable,Profound,Fun) ")
        secondary_emotion = input("Give a secondary emotion for this book: ")
        source = input("Where is the source? (Borrow,physical,PDF: ")
        language = input("What language do you plan reading it in?")
        status = input("Reading or complete: ")

        year = int(datetime.datetime.now().year)
        time_to_finish = autocompleted_dict['Time_to_finish']
        time_to_finish /= 40
        # Create a dictionary with input details
        book_details = {
            'Status': status,
            'Key_Emotion': key_emotion,
            'Secondary_E': secondary_emotion,
            'Year': year,
            'Source': source,
            'Author_Nationality': autocompleted_dict['Author_Nationality'],
            'Genre': autocompleted_dict['Genre'],
            'Topic': autocompleted_dict['Topic'],
            'Century': autocompleted_dict['Century'],
            'Time_to_finish':  time_to_finish,
            'Language': language

        }
        return book_details


if __name__ == '__main__':
    Predictor()