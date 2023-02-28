

```python
import pandas as pd
import numpy as np
import os
import webbrowser
import matrix_factorization_utilities
import pickle
```

### Data Loading


```python
raw_df = pd.read_csv('Input_Data/movie_ratings_data_set.csv')
movies_df = pd.read_csv('Input_Data/movies.csv', index_col='movie_id')
```


```python
raw_training_dataset_df = pd.read_csv('Input_Data/movie_ratings_data_set_training.csv')
raw_testing_dataset_df = pd.read_csv('Input_Data/movie_ratings_data_set_testing.csv')
```


```python
raw_df.dtypes
```




    user_id     int64
    movie_id    int64
    value       int64
    dtype: object




```python
movies_df.dtypes
```




    title    object
    genre    object
    dtype: object



### Convert to Matrix


```python
ratings_df = pd.pivot_table(raw_df, index='user_id',columns='movie_id',aggfunc=np.max)

ratings_training_df = pd.pivot_table(raw_training_dataset_df, index='user_id', columns='movie_id', aggfunc=np.max)
ratings_testing_df = pd.pivot_table(raw_testing_dataset_df, index='user_id', columns='movie_id', aggfunc=np.max)
# if one user rated the same movie more than once, take the largest rating score.
```


```python
# create html table for easy viewing
html = ratings_df.to_html(na_rep='')
```


```python
# save html to file
with open("review_matrix.html","w") as f:
    f.write(html)
```


```python
# open the file in browser
full_filename = os.path.abspath("review_matrix.html")
webbrowser.open("file://{}".format(full_filename))
```




    True



### Matrix Factorization


```python
U, M = matrix_factorization_utilities.low_rank_matrix_factorization(ratings_training_df.as_matrix(),
                                                                    num_features=11,
                                                                    regularization_amount=2)
```

    Optimization terminated successfully.
             Current function value: 552.834780
             Iterations: 908
             Function evaluations: 1355
             Gradient evaluations: 1355
    


```python
# Find all predicted ratings by multiplying the U by M
predicted_ratings = np.matmul(U, M)
```

### Measure RMSE


```python
rmse_training = matrix_factorization_utilities.RMSE(ratings_training_df.as_matrix(),
                                                    predicted_ratings)
rmse_testing = matrix_factorization_utilities.RMSE(ratings_testing_df.as_matrix(),
                                                   predicted_ratings)

print("Training RMSE: {}".format(rmse_training))
print("Testing RMSE: {}".format(rmse_testing))
```

    Training RMSE: 0.4315824818461568
    Testing RMSE: 1.227185362163823
    


```python
# Save features and predicted ratings to files for later use
pickle.dump(U, open("user_features.dat", "wb"))
pickle.dump(M, open("product_features.dat", "wb"))
pickle.dump(predicted_ratings, open("predicted_ratings.dat", "wb" ))
```

### Find Similar Products


```python
# Swap the rows and columns of product_features just so it's easier to work with
M = np.transpose(M)
```


```python
# Choose a movie to find similar movies to. Let's find movies similar to movie #5:
movie_id = 5

# Get movie #1's name and genre
movie_information = movies_df.loc[movie_id]

print("We are finding movies similar to this movie:")
print("Movie title: {}".format(movie_information.title))
print("Genre: {}".format(movie_information.genre))
```

    We are finding movies similar to this movie:
    Movie title: The Big City Judge 2
    Genre: legal drama
    


```python
# Get the features for movie #1 we found via matrix factorization
current_movie_features = M[movie_id - 1]

print("The attributes for this movie are:")
print(current_movie_features)
```

    The attributes for this movie are:
    [-0.48082573  0.69771034 -1.03627225  0.52156949  0.12221237 -2.28282266
     -0.58361372 -0.10127349  0.35200991 -0.06479319  0.35531908]
    


```python
# The main logic for finding similar movies:

# 1. Subtract the current movie's features from every other movie's features
difference = M - current_movie_features

# 2. Take the absolute value of that difference (so all numbers are positive)
absolute_difference = np.abs(difference)

# 3. Each movie has 15 features. Sum those 15 features to get a total 'difference score' for each movie
total_difference = np.sum(absolute_difference, axis=1)

# 4. Create a new column in the movie list with the difference score for each movie
movies_df['difference_score'] = total_difference

# 5. Sort the movie list by difference score, from least different to most different
sorted_movie_list = movies_df.sort_values('difference_score')

# 6. Print the result, showing the 5 most similar movies to movie_id #1
print("The five most similar movies are:")
print(sorted_movie_list[['title', 'difference_score']][0:5])
```

    The five most similar movies are:
                                title  difference_score
    movie_id                                           
    5            The Big City Judge 2          0.000000
    24           The Big City Judge 3          0.739268
    25                   Drugs & Guns          0.890776
    10        Surrounded by Zombies 1          1.198919
    27        Surrounded by Zombies 2          1.698654
    

### Make Recommendations


```python
print("Enter a user_id to get recommendations (Between 1 and 100):")
user_id_to_search = int(input())

print("Movies previously reviewed by user_id {}:".format(user_id_to_search))

reviewed_movies_df = raw_df[raw_df['user_id'] == user_id_to_search]
reviewed_movies_df = reviewed_movies_df.join(movies_df, on='movie_id')

print(reviewed_movies_df[['title', 'genre', 'value']])

input("Press enter to continue.")

print("Movies we will recommend:")

user_ratings = predicted_ratings[user_id_to_search - 1]
movies_df['rating'] = user_ratings

already_reviewed = reviewed_movies_df['movie_id']
recommended_df = movies_df[movies_df.index.isin(already_reviewed) == False]
recommended_df = recommended_df.sort_values(by=['rating'], ascending=False)

print(recommended_df[['title', 'genre', 'rating']].head(5))
```

    Enter a user_id to get recommendations (Between 1 and 100):
    12
    Movies previously reviewed by user_id 12:
                             title                  genre  value
    75              The Spy Family              spy drama      3
    76               The Sheriff 1   crime drama, western      5
    77  We Will Fight Those Aliens         sci-fi, action      5
    78            Trapped in Space        sci-fi, mystery      5
    79               The Sheriff 2   crime drama, western      5
    80    Sci-Fi Murder Detectives  supernatural, mystery      5
    Press enter to continue.
    Movies we will recommend:
                              title                          genre    rating
    movie_id                                                                
    13                The Sheriff 3           crime drama, western  4.577199
    33                 Sports Nerds                         comedy  4.577084
    34        The Serious Detective                detective drama  4.545143
    20                   Buy My App                         comedy  4.441108
    16              Master Criminal  thriller, horror, crime drama  4.433376
    
