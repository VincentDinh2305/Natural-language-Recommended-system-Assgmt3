# Import pandas library
import pandas as pd

# 1. Load digital music metadata from a compressed JSON Lines file
songs_phuong = pd.read_json('D:\Download\meta_Digital_Music.json.gz', lines=True)

# Show the first few rows of the loaded data
songs_phuong.head()

# 2. Define a dictionary to hold exploration summary of the dataset
exploration_summary_corrected = {
    "Column": [],
    "Data Type": [],
    "Null Count": [],
    "Empty String Count": [],
    "Empty List Count": [],
    "Unique Values or Note": []
}

# Loop through each column in the dataframe
for column in songs_phuong.columns:
    # Append column name, data type, and null count to respective lists
    exploration_summary_corrected["Column"].append(column)
    exploration_summary_corrected["Data Type"].append(songs_phuong[column].dtype)
    exploration_summary_corrected["Null Count"].append(songs_phuong[column].isnull().sum())
    
    # Count empty strings and lists in object-type columns
    if songs_phuong[column].dtype == 'object':
        empty_string_count = (songs_phuong[column] == '').sum()
        empty_list_count = songs_phuong[column].apply(lambda x: x == [] if isinstance(x, list) else False).sum()
    else:
        empty_string_count = 0
        empty_list_count = 0
    
    exploration_summary_corrected["Empty String Count"].append(empty_string_count)
    exploration_summary_corrected["Empty List Count"].append(empty_list_count)
    
    # Calculate unique values or note unhashable types
    try:
        unique_values = songs_phuong[column].nunique()
    except TypeError:  # Handle unhashable types
        unique_values = "Contains unhashable types, unique count skipped"
    
    exploration_summary_corrected["Unique Values or Note"].append(unique_values)

# Convert the summary dictionary into a DataFrame for easy viewing
exploration_df_corrected = pd.DataFrame(exploration_summary_corrected)
# Display the resulting DataFrame
exploration_df_corrected

# Drop irrelevant or heavily null columns from the dataframe
columns_to_drop = ['tech1', 'tech2', 'fit', 'similar_item', 'date', 'imageURLHighRes']
songs_phuong_filtered = songs_phuong.drop(columns=columns_to_drop)

# Remove rows missing critical information ('title', 'brand', 'asin')
songs_phuong_filtered = songs_phuong_filtered[(songs_phuong_filtered['title'] != '') &
                                               (songs_phuong_filtered['brand'] != '') &
                                               (songs_phuong_filtered['asin'] != '')]

# Display the first few rows of the filtered dataframe
songs_phuong_filtered.head()

# 3. Feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# a. Combine 'title' and 'description' for better text analysis. Missing descriptions are handled as empty strings.
songs_phuong_filtered['combined_text'] = songs_phuong_filtered['title'] + " " 
+ songs_phuong_filtered['description'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')

# b. Initialize TfidfVectorizer, excluding English stop words for text preprocessing and vectorization.
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Create TF-IDF vectors for combined text.
tfidf_matrix = tfidf_vectorizer.fit_transform(songs_phuong_filtered['combined_text'])

# c. Compute cosine similarity across all TF-IDF vectors to measure song similarities.
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Define a function to recommend songs based on a given title using cosine similarity scores.
def get_recommendations(title, cosine_sim=cosine_sim):
    # Match the input title to dataset titles, compute similarity scores, and retrieve top 10 similar titles.
    idx = songs_phuong_filtered.index[songs_phuong_filtered['title'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    song_indices = [i[0] for i in sim_scores]
    return songs_phuong_filtered['title'].iloc[song_indices]

# Example usage: Get and save recommendations for the first song in the filtered dataset.
first_song_title = songs_phuong_filtered['title'].iloc[0]
recommendations = get_recommendations(first_song_title)

# d. Save the song recommendation
recommendations.to_csv('D:\Download\song_recommendations.csv', index=False)

recommendations

# 4. Song Recommender
def recommend_songs_interactively():
    while True:
        # Ask the user for a song title or 'exit' command
        user_input = input("Enter a song title for recommendations or type 'exit' to quit: ")

        # Stop the loop if user decides to exit
        if user_input.lower() == 'exit':
            print("Exiting the recommender system. Goodbye!")
            break

        # If the song exists in our dataset, display recommendations
        if user_input in songs_phuong_filtered['title'].values:
            recommendations = get_recommendations(user_input)  # Fetch recommendations
            print("Top recommendations for '{}':".format(user_input))
            for idx, rec in enumerate(recommendations, start=1):  # Enumerate through recommendations and print
                print("{}. {}".format(idx, rec))
        else:
            # Let the user know if the song is not in the dataset
            print("We don’t have recommendations for {}".format(user_input))

# Call the function to start the interactive recommendation session
recommend_songs_interactively()

