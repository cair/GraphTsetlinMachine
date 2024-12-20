import pandas as pd
import kagglehub
import numpy as np

np.random.seed(42)

def amazon_products():
    print("Creating training data")
    path = kagglehub.dataset_download("karkavelrajaj/amazon-sales-dataset")
    print("Path to dataset files:", path)
    data_file = path + "/amazon.csv" 
    org_data = pd.read_csv(data_file)
    print("Original data shape:", org_data.shape)
    return org_data[['product_id', 'category', 'user_id', 'rating']]

def aug_amazon_products():
    org_data = amazon_products()
    org_data['rating'] = pd.to_numeric(org_data['rating'], errors='coerce')  # Coerce invalid values to NaN
    org_data.dropna(subset=['rating'], inplace=True)  # Drop rows with NaN ratings
    org_data['rating'] = org_data['rating'].astype(int)
    # Expand the dataset 10 times
    data = pd.concat([org_data] * 10, ignore_index=True)
    # Shuffle the expanded dataset
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    # Add noise
    # Define the noise ratio
    noise_ratio = 0.1  # 10% noise
    # Select rows to apply noise
    num_noisy_rows = int(noise_ratio * len(data))
    noisy_indices = np.random.choice(data.index, size=num_noisy_rows, replace=False)
    # Add noise to ratings
    data.loc[noisy_indices, 'rating'] = np.random.choice(range(1, 6), size=num_noisy_rows)
    # Add noise to categories
    unique_categories = data['category'].unique()
    data.loc[noisy_indices, 'category'] = np.random.choice(unique_categories, size=num_noisy_rows)
    # Print a preview of the noisy and expanded dataset
    print("Expanded data shape:", data.shape)
    print("Data preview:\n", data.head())
    return data

def artificial():
    num_users = 5  # Number of unique users
    num_items =10  # Number of unique items
    num_categories = 5  # Number of unique categories
    num_interactions = 1000  # Number of user-item interactions
    # Generate random ratings (e.g., between 1 and 5)
    ratings = np.random.choice(range(1, 3), num_interactions)
    # Generate random user-item interactions
    user_ids = np.random.choice(range(num_users), num_interactions)
    item_ids = np.random.choice(range(num_items), num_interactions)
    categories = np.random.choice(range(num_categories), num_interactions)

    data = pd.DataFrame({
        'user_id': user_ids,
        'product_id': item_ids,
        'category': categories,
        'rating': ratings
    })
    return data

def artificial_with_user_pref():
    num_users = 100  # Number of unique users
    num_items = 50    # Number of unique items
    num_categories = 50  # Number of unique categories
    num_interactions = 1000  # Number of user-item interactions
    noise_ratio = 0.01  # Percentage of noisy interactions

    # Generate user preferences: each user prefers 1-3 random categories
    user_preferences = {
        user: np.random.choice(range(num_categories), size=np.random.randint(1, 4), replace=False)
        for user in range(num_users)
    }

    # Assign each item to a category
    item_categories = {item: np.random.choice(range(num_categories)) for item in range(num_items)}

    # Generate interactions
    user_ids = np.random.choice(range(num_users), num_interactions)
    item_ids = np.random.choice(range(num_items), num_interactions)

    # Generate ratings based on the pattern
    ratings = []
    for user, item in zip(user_ids, item_ids):
        item_category = item_categories[item]
        if item_category in user_preferences[user]:
            ratings.append(np.random.choice([3, 4]))  # High rating for preferred categories
        else:
            ratings.append(np.random.choice([1, 2]))  # Low rating otherwise

    # Introduce noise
    num_noisy = int(noise_ratio * num_interactions)
    noisy_indices = np.random.choice(range(num_interactions), num_noisy, replace=False)
    for idx in noisy_indices:
        ratings[idx] = np.random.choice(range(1, 6))  # Replace with random rating

    # Combine into a DataFrame
    data = pd.DataFrame({
        'user_id': user_ids,
        'product_id': item_ids,
        'category': [item_categories[item] for item in item_ids],
        'rating': ratings
    })
    return data

def artificial_pattered():
    num_users = 100 # Number of unique users
    num_items = 50    # Number of unique items
    num_categories = 5  # Number of unique categories
    num_interactions = 10000  # Number of user-item interactions
    noise_ratio = 0.01  # Percentage of noisy interactions

    # Step 1: Define deterministic user preferences
    user_preferences = {user: user % num_categories for user in range(num_users)}

    # Step 2: Assign items to categories in a cyclic pattern
    item_categories = {item: item % num_categories for item in range(num_items)}

    # Step 3: Generate deterministic interactions
    user_ids = np.arange(num_interactions) % num_users  # Cycle through users
    item_ids = np.arange(num_interactions) % num_items  # Cycle through items

    # Step 4: Generate ratings based on the pattern
    ratings = []
    for user, item in zip(user_ids, item_ids):
        preferred_category = user_preferences[user]
        item_category = item_categories[item]
        if item_category == preferred_category:
            ratings.append(5)  # High rating for preferred category
        else:
            ratings.append(1)  # Low rating otherwise

    # Step 5: Introduce noise
    num_noisy = int(noise_ratio * num_interactions)
    noisy_indices = np.random.choice(range(num_interactions), num_noisy, replace=False)
    for idx in noisy_indices:
        ratings[idx] = np.random.choice(range(1, 6))  # Replace with random rating

    # Step 6: Create a DataFrame
    data = pd.DataFrame({
        'user_id': user_ids,
        'product_id': item_ids,
        'category': [item_categories[item] for item in item_ids],
        'rating': ratings
    })
    return data