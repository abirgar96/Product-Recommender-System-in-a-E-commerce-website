# main.py
# from . import chat
from random import random
import chat
from autocorrect import Speller
from flask import Flask,Blueprint, render_template, request,jsonify
from flask_login import login_required, current_user
from datetime import datetime
import os
import json
import random

from flask_cors import CORS,cross_origin
import pandas as pd
import numpy as np
import webbrowser
import matrix_factorization_utilities
import pickle



###################################################################################

app = Flask(__name__)
main = Blueprint('main', __name__)
# CORS(app,ressources={r"/*":{"origins":"*"}})
CORS(app)
# cors = CORS(app, resources={r"/rec": {"origins": "http://localhost:5000"}})

################################# Rec sys #########################################
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

# Save features and predicted ratings to files for later use
pickle.dump(U, open("res1/user_features.dat", "wb"))
pickle.dump(M, open("res1/product_features.dat", "wb"))
pickle.dump(predicted_ratings, open("res1/predicted_ratings.dat", "wb" ))

# ### Find Similar Products
# Swap the rows and columns of product_features just so it's easier to work with
M = np.transpose(M)




@app.route("/get/<int:user_id_to_search>", methods=['GET'])
def get_rec(user_id_to_search):
    if request.method =='GET':
        # content = request.json
        # user_id_to_search=content["userid"]



# @app.route("/rec/<user_id_to_search>", methods=['GET','POST'])
# @cross_origin()
# def get_rec(user_id_to_search):
#     if request.method =='POST':


        #content = request.json
        # user_id_to_search=content["userid"]
        # Choose a product to find similar products to. Let's find movies similar to product #5:
        print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh ",user_id_to_search)
        df=pd.read_csv('Input_Data/product_ratings_data_set.csv')

        newdf = df[(df.user_id == user_id_to_search) ]
        newdf = newdf[(df.value >4)]

        liste_product_id=newdf["product_id"].tolist()
        print("lissssst after filter ",liste_product_id)

        product_id = random.choice(liste_product_id)
        # Get product #1's name and genre
        product_information = products_df.loc[product_id]

        # Get the features for product #1 we found via matrix factorization
        current_product_features = M[product_id - 1]

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
        reviewed_products_df = raw_df[raw_df['user_id'] == user_id_to_search]
        reviewed_products_df = reviewed_products_df.join(products_df, on='product_id')
        user_ratings = predicted_ratings[user_id_to_search - 1]
        products_df['rating'] = user_ratings

        already_reviewed = reviewed_products_df['product_id']
        recommended_df = products_df[products_df.index.isin(already_reviewed) == False]
        recommended_df = recommended_df.sort_values(by=['rating'], ascending=False)
        print('hhhhhhhhhhhhhhhhhhhhhh ',recommended_df)
        x=recommended_df[['name', 'category', 'rating']].values
        
        y=x.tolist()
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx ',y)

        recom=[]
        for name,category,rating in y:
            # if rating >4:
                recom.append({"product Id":product_id,"name":name,"category":category,"rating":rating})
        
        res={"recommandation":recom}
        print("lissssst after recom ",recom)

        return jsonify(res)
    response="hello"
    return {"msg":response}


###################################################################################

@app.route("/chat", methods=['GET','POST'])
def get_bot_response():
    spell = Speller()
    userText = request.args.get('msg')
    if request.method =='POST':
        content = request.json
        userText=content["usertext"]
        response= str(chat.chatbot_response(userText))
        res={"rep":str(response)}
        return jsonify(res)
    response= str(chat.chatbot_response(userText))
    if response in ["Sorry, can't understand you", "Please give me more info", "Not sure I understand"]:
        response= str(chat.chatbot_response(spell(userText)))

    return {"msg":str(response)}




if __name__ == "__main__":
    app.run(host="localhost",debug=True)
