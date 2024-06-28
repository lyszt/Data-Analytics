import datetime
import json

import openai
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle



class Predictor:

    def __init__(self):
        self.model = None
        
        self.Train()
        # Load LabelEncoders
        label_encoders = self.load_label_encoders()

        # Get book details from user input
        book_details = self.input_book_details()

        # Predict if you'll like the book
        prediction = self.predict_liking(book_details, label_encoders)

        # Print prediction result
        if prediction[0] == 1:
            print("You'll like this book!")
        else:
            print("You might not like this book.")
    def Train(self):
        # Load data
        df = pd.read_csv('logs.csv',encoding='latin-1')
        df = df.fillna(0)

        # Preprocessing
        df['Time_to_finish'] = (df['Time_to_finish']).fillna(0)
        df['Pages'] = (df['Pages']).fillna(0)
        df['Pages_Read'] = (df['Pages_Read']).fillna(0)
        df['Century'] = (df['Century']).fillna(0)

        df['Year'] = df['Year'].astype(int)
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

        df['Author_nationality_encoded'] = label_encoder.fit_transform(df['Author_Nationality'])
        df['Genre_encoded'] = label_encoder.fit_transform(df['Genre'])
        df['Language_encoded'] = label_encoder.fit_transform(df['Language'])
        df['Topic_encoded'] = label_encoder.fit_transform(df['Topic'])
        df['Source_encoded'] = label_encoder.fit_transform(df['Source'])
        df['Status_encoded'] = label_encoder.fit_transform(df['Status'])
        df['Year_encoded'] = label_encoder.fit_transform(df['Year'])
        df['Keyemotion_encoded'] = label_encoder.fit_transform(df['Key_Emotion'])
        df['Secemotion_encoded'] = label_encoder.fit_transform(df['Secondary_E'])

        # Save LabelEncoder objects to files
        label_encoders = {
            'Author_nationality': label_encoder,
            'Genre': label_encoder,
            'Language': label_encoder,
            'Topic': label_encoder,
            'Source': label_encoder,
            'Status': label_encoder,
            'Year': label_encoder,
            'Key_Emotion': label_encoder,
            'Secondary_E': label_encoder
        }

        with open('label_encoders.pkl', 'wb') as le_file:
            pickle.dump(label_encoders, le_file)

        # Features and target variable
        features = df[['Author_nationality_encoded', 'Genre_encoded', 'Language_encoded', 'Topic_encoded',
                       'Source_encoded', 'Status_encoded', 'Keyemotion_encoded', 'Secemotion_encoded', 'Time_to_finish','Year_encoded']]
        target = df['Liked']
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        # Train a Random Forest classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        # Make predictions
        y_pred = self.model.predict(X_test)
        # Evaluate the self.model
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
    def load_label_encoders(self,filename='label_encoders.pkl'):
        with open(filename, 'rb') as le_file:
            label_encoders = pickle.load(le_file)
        return label_encoders
    def predict_liking(self,book_details, label_encoders):
        # Load LabelEncoders from file
        author_nationality_encoder = label_encoders['Author_nationality']
        genre_encoder = label_encoders['Genre']
        language_encoder = label_encoders['Language']
        topic_encoder = label_encoders['Topic']
        source_encoder = label_encoders['Source']
        status_encoder = label_encoders['Status']
        year_encoder = label_encoders['Year']
        keyemotion_encoder = label_encoders['Key_Emotion']
        secemotion_encoder = label_encoders['Secondary_E']

        # Convert input details to DataFrame
        book_df = pd.DataFrame([book_details])

        # Encode categorical variables using loaded LabelEncoders
        try:
            book_df['Author_nationality_encoded'] = author_nationality_encoder.transform(book_df['Author_nationality'].astype(str))
        except ValueError:
            # Handle unseen label for Author Nationality
            # Example fallback: Assign a default value or handle it based on your domain knowledge
            book_df['Author_nationality_encoded'] = -1  # Assigning a default value

        try:
            book_df['Genre_encoded'] = genre_encoder.transform(book_df['Genre'].astype(str))
        except ValueError:
            book_df['Genre_encoded'] = -1  # Assigning a default value

        try:
            book_df['Language_encoded'] = language_encoder.transform(book_df['Language'].astype(str))
        except ValueError:
            book_df['Language_encoded'] = -1  # Assigning a default value

        try:
            book_df['Topic_encoded'] = topic_encoder.transform(book_df['Topic'].astype(str))
        except ValueError:
            book_df['Topic_encoded'] = -1  # Assigning a default value

        try:
            book_df['Source_encoded'] = source_encoder.transform(book_df['Source'].astype(str))
        except ValueError:
            book_df['Source_encoded'] = -1  # Assigning a default value

        try:
            book_df['Status_encoded'] = status_encoder.transform(book_df['Status'].astype(str))
        except ValueError:
            book_df['Status_encoded'] = -1  # Assigning a default value

        try:
            book_df['Year_encoded'] = year_encoder.transform(book_df['Year'].astype(int))
        except ValueError:
            book_df['Year_encoded'] = -1  # Assigning a default value

        try:
            book_df['Keyemotion_encoded'] = keyemotion_encoder.transform(book_df['Key_Emotion'].astype(str))
        except ValueError:
            book_df['Keyemotion_encoded'] = -1  # Assigning a default value

        try:
            book_df['Secemotion_encoded'] = secemotion_encoder.transform(book_df['Secondary_E'].astype(str))
        except ValueError:
            book_df['Secemotion_encoded'] = -1  # Assigning a default value

        # Extract features from input DataFrame
        features_input = book_df[['Author_nationality_encoded', 'Genre_encoded', 'Language_encoded',
                                  'Topic_encoded', 'Source_encoded', 'Status_encoded', 'Keyemotion_encoded',
                                  'Secemotion_encoded', 'Time_to_finish', 'Year_encoded']]

        # Make prediction using trained self.model
        prediction = self.model.predict(features_input)

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
                                    "rules.\nPossible inputs:\n\"Author_nationality\" can ONLY equal European, Brazilian, "
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
            'Author_nationality': autocompleted_dict['Author_Nationality'],
            'Genre': autocompleted_dict['Genre'],
            'Topic': autocompleted_dict['Topic'],
            'Century': autocompleted_dict['Century'],
            'Time_to_finish':  time_to_finish,
            'Language': language

        }
        return book_details


if __name__ == '__main__':
    Predictor()