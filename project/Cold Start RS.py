
import pandas as pd
import numpy as np
import matrix_factorization_utilities
import pickle
### Data Loading

raw_df = pd.read_csv('Input_Data/product_ratings_data_set.csv')

## Convert to Matrix

ratings_df = pd.pivot_table(raw_df, index='user_id',columns='product_id',aggfunc=np.max)
# if one user rated the same product more than once, take the largest rating score.
## Rating Score Normalization

# Normalize the ratings (center them around their mean)
# normalized_ratings, means = matrix_factorization_utilities.normalize_ratings(ratings_df.as_matrix())
normalized_ratings, means = matrix_factorization_utilities.normalize_ratings(ratings_df.as_matrix())

### Matrix Factorization

# Apply matrix factorization to find the latent features
U, M = matrix_factorization_utilities.low_rank_matrix_factorization(normalized_ratings,
                                                                    num_features=11,
                                                                    regularization_amount=1.1)

# Find all predicted ratings by multiplying U and M
predicted_ratings = np.matmul(U, M)

# Add back in the mean ratings for each product to de-normalize the predicted results
predicted_ratings = predicted_ratings + means


# Save features and predicted ratings to files for later use
pickle.dump(U, open("res1/user_features.dat", "wb"))
pickle.dump(M, open("res1/product_features.dat", "wb"))
pickle.dump(predicted_ratings, open("res1/predicted_ratings.dat", "wb" ))
pickle.dump(means, open("res1/means.dat", "wb" ))




