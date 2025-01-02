import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import folium

def load_and_preprocess_data(file_path):
    # Dataset Loading
    data = pd.read_csv(file_path, low_memory=False)
    data = data[['DBA', 'ZIPCODE', 'GRADE', 'SCORE', 'Latitude', 'Longitude']].drop_duplicates().dropna(subset=['DBA', 'ZIPCODE'])
    data['ZIPCODE'] = data['ZIPCODE'].apply(clean_zipcode)  # ZIPCODE cleaning
    data['zipcode_encoded'] = data['ZIPCODE'].astype('category').cat.codes
    data['Star_Rating'] = data.apply(calculate_star_rating, axis=1)
    print(f"Data loaded with {len(data)} entries.")
    return data

def clean_zipcode(zipcode):
    """Convert ZIP codes to strings and remove any trailing '.0'."""
    if pd.notna(zipcode):
        zipcode = str(zipcode).split('.')[0]
    return zipcode

def calculate_star_rating(row):
    if pd.notna(row['GRADE']):
        return grade_to_star(row['GRADE'])
    elif pd.notna(row['SCORE']):
        return score_to_star(row['SCORE'])
    return None

def grade_to_star(grade):
    mapping = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}
    return mapping.get(grade, None)

def score_to_star(score):
    if score <= 13:
        return 5
    elif score <= 27:
        return 4
    elif score <= 41:
        return 3
    elif score <= 55:
        return 2
    return 1

def prepare_model_data(data):
    data = data.dropna(subset=['Star_Rating']).copy()
    data['zipcode_encoded'] = data['ZIPCODE'].astype('category').cat.codes
    X = data[['zipcode_encoded']]
    y = (data['Star_Rating'] >= 4).astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    param_grid = {'n_estimators': [100, 200], 'max_depth': [5, 10], 'random_state': [42]}
    model = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    model.fit(X_train, y_train)
    print(f"Best Parameters: {model.best_params_}")
    return model.best_estimator_

def predict_best_restaurants(model, data, postal_code, top_n=5):
    postal_code_data = data[data['ZIPCODE'] == postal_code].copy()
    print(f"Found {len(postal_code_data)} entries for postal code {postal_code}")  # Debugging 

    if postal_code_data.empty:
        print(f"No data available for postal code {postal_code}")
        return

    postal_code_data['prediction'] = model.predict(postal_code_data[['zipcode_encoded']])
    best_restaurants = postal_code_data[postal_code_data['prediction'] == 1].sort_values(by='Star_Rating', ascending=False).head(top_n)
    if not best_restaurants.empty:
        print(f"Found {len(best_restaurants)} recommendations for postal code {postal_code}.")
    display_restaurants_on_map(best_restaurants)

def display_restaurants_on_map(restaurants):
    if restaurants.empty:
        print("No recommended restaurants to display on the map.")
        return
    map_center = [restaurants.iloc[0]['Latitude'], restaurants.iloc[0]['Longitude']]
    map = folium.Map(location=map_center, zoom_start=15)
    for _, row in restaurants.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"{row['DBA']} - Rating: {row['Star_Rating']}",
            icon=folium.Icon(color='green')
        ).add_to(map)
    map.save('recommended_restaurants_map.html')
    print("Map of recommended restaurants has been saved to 'recommended_restaurants_map.html'.")

def visualize_data(data, model_accuracy):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=data['Star_Rating'], palette='viridis')
    plt.title("Distribution of Star Ratings")
    plt.xlabel("Star Rating")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.bar(["Accuracy"], [model_accuracy], color='skyblue')
    plt.ylim(0, 1)
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.show()

def main():
    file_path = 'DOHMH_New_York_City_Restaurant_Inspection_Results_20231130.csv'  # Placeholder
    data = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test = prepare_model_data(data)
    model = train_model(X_train, y_train)
    model_accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model Accuracy: {model_accuracy}")

    visualize_data(data, model_accuracy)

    while True:
        user_postal_code = input("Enter your postal code (or type 'exit' to quit): ")
        print(f"User entered postal code: {user_postal_code}")  # Debugging 
        if user_postal_code.lower() == 'exit':
            print("Exiting...")
            break
        predict_best_restaurants(model, data, user_postal_code)

if __name__ == "__main__":
    main()
