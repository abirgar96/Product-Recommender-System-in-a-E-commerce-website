
import pandas as pd
import numpy as np
import os
import webbrowser
import matrix_factorization_utilities
import pickle
from flask import Flask, request,jsonify,Blueprint
import random

app = Flask(__name__)
# main = Blueprint('main', __name__)

# ### Data Loading

raw_df = pd.read_csv('Input_Data/product_ratings_data_set.csv')
products_df = pd.read_csv('Input_Data/abir.csv', index_col='product_id')

raw_training_dataset_df = pd.read_csv('Input_Data/product_ratings_data_set_training.csv')
raw_testing_dataset_df = pd.read_csv('Input_Data/product_ratings_data_set_testing.csv')


## Convert to Matrix

ratings_df = pd.pivot_table(raw_df, index='user_id',columns='product_id',aggfunc=np.max)

ratings_training_df = pd.pivot_table(raw_training_dataset_df, index='user_id', columns='product_id', aggfunc=np.max)
ratings_testing_df = pd.pivot_table(raw_testing_dataset_df, index='user_id', columns='product_id', aggfunc=np.max)
# if one user rated the same movie more than once, take the largest rating score.

# create html table for easy viewing
html = ratings_df.to_html(na_rep='')

# # save html to file
# with open("res/review_matrix.html","w") as f:
#     f.write(html)


## Matrix Factorization

U, M = matrix_factorization_utilities.low_rank_matrix_factorization(ratings_training_df.as_matrix(),
                                                                    num_features=11,
                                                                    regularization_amount=2)

# Find all predicted ratings by multiplying the U by M
predicted_ratings = np.matmul(U, M)

# ### Measure RMSE

rmse_training = matrix_factorization_utilities.RMSE(ratings_training_df.as_matrix(),
                                                    predicted_ratings)
rmse_testing = matrix_factorization_utilities.RMSE(ratings_testing_df.as_matrix(),
                                                   predicted_ratings)

print("Training RMSE: {}".format(rmse_training))
print("Testing RMSE: {}".format(rmse_testing))

# Save features and predicted ratings to files for later use
pickle.dump(U, open("res1/user_features.dat", "wb"))
pickle.dump(M, open("res1/product_features.dat", "wb"))
pickle.dump(predicted_ratings, open("res1/predicted_ratings.dat", "wb" ))

# ### Find Similar Products

# Swap the rows and columns of product_features just so it's easier to work with
M = np.transpose(M)

# Choose a product to find similar products to. Let's find products similar to product #5:
product_id = 5
#product_id = random.choice["product_id"].tolist()

# Get product #1's name and genre
product_information = products_df.loc[product_id]

print("We are finding products similar to this product:")
print("Product title: {}".format(product_information.name))
print("Genre: {}".format(product_information.category))

# Get the features for product #1 we found via matrix factorization
current_product_features = M[product_id - 1]

print("The attributes for this product are:")
print(current_product_features)

# The main logic for finding similar products:

# 1. Subtract the current product's features from every other product's features
difference = M - current_product_features

# 2. Take the absolute value of that difference (so all numbers are positive)
absolute_difference = np.abs(difference)

# 3. Each product has 15 features. Sum those 15 features to get a total 'difference score' for each movie
total_difference = np.sum(absolute_difference, axis=1)

# 4. Create a new column in the product list with the difference score for each product
products_df['difference_score'] = total_difference

# 5. Sort the product list by difference score, from least different to most different
sorted_product_list = products_df.sort_values('difference_score')

# 6. Print the result, showing the 5 most similar products to product_id #1
print("The five most similar products are:")
print(sorted_product_list[['name', 'difference_score']][0:5])

## Make Recommendations


@app.route("/get", methods=['GET','POST'])
def get_rec():
    userText = request.args.get('msg')
    if request.method =='POST':
        content = request.json
        user_id_to_search=content["userid"]
        reviewed_products_df = raw_df[raw_df['user_id'] == user_id_to_search]
        reviewed_products_df = reviewed_products_df.join(products_df, on='product_id')
        user_ratings = predicted_ratings[user_id_to_search - 1]
        products_df['rating'] = user_ratings

        already_reviewed = reviewed_products_df['product_id']
        recommended_df = products_df[products_df.index.isin(already_reviewed) == False]
        recommended_df = recommended_df.sort_values(by=['rating'], ascending=False)
        x=recommended_df[['name', 'category', 'rating']].values
        y=x.tolist()
        print("yyyyyyyyyyyyyyyyyyyyyy ",y)

        recom=[]
        for name,category,rating in y:
            recom.append({"product Id":555,"name":name,"category":category,"rating":rating})
        
        res={"recommandation":recom}
        return jsonify(res)
    response="hello"
    return {"msg":response}

if __name__ == "__main__":
   app.run(debug=True)

