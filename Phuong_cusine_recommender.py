# Import necessary library
import json
import pandas as pd
from apyori import apriori
from collections import Counter

# 1. Load the data

# File path
file_path = r'D:\Download\recipes.json'
with open(file_path, 'r') as file:
    data = json.load(file)
    
# Convert the data into a DataFrame for easier analysis
recipes = pd.DataFrame(data)

# Note the total number of instances
total_instances = len(recipes)

# Note the number of cuisines available in the data
unique_cuisines = recipes['cuisine'].nunique()

# Create a table illustrating each cuisine type and # of recipes available in the file related to that cuisine
cuisine_counts = recipes['cuisine'].value_counts().reset_index()
cuisine_counts.columns = ['Cuisine Type', 'Number of Recipes']

print(f"Total number of instances (recipes): {total_instances}")
print(f"Number of unique cuisines: {unique_cuisines}")
print("Table of each cuisine type and the number of recipes available:")
print(cuisine_counts)

# Average Number of Ingredients per Cuisine
recipes['ingredient_count'] = recipes['ingredients'].apply(len)
average_ingredients = recipes.groupby('cuisine')['ingredient_count'].mean().reset_index()
average_ingredients.columns = ['Cuisine Type', 'Average Number of Ingredients']
print(average_ingredients)

# Most Common Ingredients Overall

# Flatten the list of ingredients from all recipes
all_ingredients = [ingredient for sublist in recipes['ingredients'] for ingredient in sublist]

# Count the occurrences of each ingredient
ingredient_counts = Counter(all_ingredients)

# Most common ingredients
most_common_ingredients = pd.DataFrame(ingredient_counts.most_common(), columns=['Ingredient', 'Frequency'])
print("Most common ingredients overall:")
print(most_common_ingredients.head(10))

# Define the function to find most common ingredients by cuisine
def most_common_ingredients_by_cuisine(cuisine, n=5):
    """Function to print the most common ingredients for a given cuisine."""
    cuisine_data = recipes[recipes['cuisine'] == cuisine]
    cuisine_ingredients = [ingredient for sublist in cuisine_data['ingredients'] for ingredient in sublist]
    ingredient_counts = Counter(cuisine_ingredients)
    most_common = pd.DataFrame(ingredient_counts.most_common(n), columns=['Ingredient', 'Frequency'])
    print(f"Most common ingredients for {cuisine}:")
    print(most_common)
    print("\n")

# Loop through each unique cuisine and apply the function
for cuisine in recipes['cuisine'].unique():
    most_common_ingredients_by_cuisine(cuisine, 10)
    
def apriori_analysis(cuisine, recipes):
    # Filter recipes for the selected cuisine
    cuisine_recipes = recipes[recipes['cuisine'].str.lower() == cuisine.lower()]
    if len(cuisine_recipes) == 0:
        return None, None

    print(f"Analyzing {len(cuisine_recipes)} recipes for cuisine: {cuisine}")
    
    # Prepare transactions for Apriori
    transactions = list(cuisine_recipes['ingredients'])
    support_value = 100 / len(cuisine_recipes)
    print(f"Using support value: {support_value/100:.4f}")

    # Run Apriori analysis
    results = list(apriori(transactions, min_support=support_value/100, min_confidence=0.46, min_lift=2, min_length=2))

    # Extract top 2 ingredient combinations by their support values
    top_ingredients_sets = sorted(results, key=lambda x: x.support, reverse=True)[:2]
    
    # Extract rules with lift > 2
    rules_with_lift_over_2 = []
    for result in results:
        for rule in result.ordered_statistics:
            if rule.lift > 2:
                rules_with_lift_over_2.append(rule)

    # Format top ingredient sets for display
    top_sets_formatted = []
    for top_set in top_ingredients_sets:
        items = ', '.join(top_set.items)
        support = top_set.support
        top_sets_formatted.append(f"Ingredients: {items} | Support: {support:.4f}")

    # Format rules for display
    rules_formatted = []
    for rule in rules_with_lift_over_2:
        base = ', '.join(rule.items_base)
        add = ', '.join(rule.items_add)
        confidence = rule.confidence
        lift = rule.lift
        rules_formatted.append(f"Rule: {base} -> {add} | Confidence: {confidence:.4f} | Lift: {lift:.2f}")

    return top_sets_formatted, rules_formatted

# Main interaction loop
while True:
    cuisine_input = input("Please enter a cuisine type (or 'bye' to exit): ").strip().lower()
    
    if cuisine_input == 'bye':
        print("Goodbye!")
        break
    
    if cuisine_input not in recipes['cuisine'].str.lower().unique():
        print(f"We don’t have recommendations for {cuisine_input.title()}")
        continue

    top_ingredients_sets, rules_with_lift_over_2 = apriori_analysis(cuisine_input, recipes)
    
    if top_ingredients_sets is None or rules_with_lift_over_2 is None:
        print(f"We don’t have recommendations for {cuisine_input.title()}")
        continue

    print("\nTop ingredient combinations:")
    for combination in top_ingredients_sets:
        print(combination)
    
    print("\nRules with lift > 2:")
    for rule in rules_with_lift_over_2:
        print(rule)

    print("\n")