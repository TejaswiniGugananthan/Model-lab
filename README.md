```python
3. Implementation Of GSP Algorithm in Python
 # Function to generate candidate k-item sequences
 def generate_candidates(dataset, k, min_support):
 candidates = defaultdict(int)
 for seq in dataset:
 for comb in combinations(seq, k):
 candidates[comb] += 1
 return {item: support for item, support in candidates.items() if support >= min_support}
 # Function to perform GSP algorithm
 def gsp(dataset, min_support):
 frequent_patterns = defaultdict(int) # Corrected variable name
 k = 1
 sequences = dataset
 while True:
 candidates = generate_candidates(sequences, k, min_support)
 if not candidates:
 break
 frequent_patterns.update(candidates)
 k += 1
 return frequent_patterns


4. Implementation of Cluster and visitor segmentation for Navigation patterns
 # Visitor segmentation based on characteristics
 import pandas as pd
 import matplotlib.pyplot as plt
# read the data
 visitor_df=pd.read_csv('/content/clustervisitor (1).csv')
# Perform segmentation based on characteristics (e.g., age groups)
 age_groups={
 'Young':(visitor_df['Age']<=30),
 'Middle-aged':((visitor_df['Age']>30) & (visitor_df['Age'] <=50)),
 'Eldery':(visitor_df['Age'] > 50)
 }
 for group,condition in age_groups.items():
 visitors_in_group=visitor_df[condition]
 print(f"Visitors in {group} age group")
 print(visitors_in_group)
 print()
# Visualization:
# Create a list to store counts of visitors in each age group
 visitor_counts = []
# Count visitors in each age group
 for group, condition in age_groups.items():
 count = visitor_df[condition].shape[0]
 visitor_counts.append(count)
# Define age group labels
 age_group_labels = list(age_groups.keys())
# Plot a bar chart
 import matplotlib.pyplot as plt


5.Information Retrieval Using Boolean Model in Python
def boolean_search(self, query):
    query_terms = query.lower().split()
    results = set() 
    current_set = None 
    i = 0
    while i < len(query_terms):
        term = query_terms[i]
        if term == 'or':
            if current_set is not None:
                results.update(current_set)
                current_set = None 
            elif term == 'and':
                i += 1
                continue 
            elif term == 'not':
                i += 1
                if i < len(query_terms):
                    not_term = query_terms[i]
                    if not_term in self.index:
                        not_docs = self.index[not_term]
                        if current_set is None:
                            current_set = set(range(1, len(documents) + 1)) 
                        current_set.difference_update(not_docs)
        else:
            if term in self.index:
                term_docs = self.index[term]
                if current_set is None:
                    current_set = term_docs.copy()
                else:
                    current_set.intersection_update(term_docs)
            else:
                current_set = set() 
        i += 1
 # Update results with the last processed set
 if current_set is not None:
     results.update(current_set)
 return sorted(results)



6. Information Retrieval Using Vector Space Model in Python
 # Calculate cosine similarity between query and documents
 def search(query, tfidf_matrix, tfidf_vectorizer):
     preprocessed_query = preprocess_text(query)
     query_vector = tfidf_vectorizer.transform([preprocessed_query])
 # Calculate cosine similarity between query and documents
 similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
 # Sort documents based on similarity scores
 sorted_indexes = similarity_scores.argsort()[0][::-1]
 # Return sorted documents along with their similarity scores and original indexes
 results = [(documents[i], similarity_scores[0, i], i + 1) for i in sorted_indexes] 
 return results




7.Implementation of Link Analysis using HITS Algorithm
 num_nodes = len(adjacency_matrix)
 authority_scores = np.ones(num_nodes)
 hub_scores = np.ones(num_nodes)
 # Authority update
 new_authority_scores = np.dot(adjacency_matrix.T, hub_scores)
 new_authority_scores /= np.linalg.norm(new_authority_scores, ord=2) # Normalizing
 # Hub update
 new_hub_scores = np.dot(adjacency_matrix, new_authority_scores)
 new_hub_scores /= np.linalg.norm(new_hub_scores, ord=2) # Normalizing
 # Check convergence
 authority_diff = np.linalg.norm(new_authority_scores- authority_scores, ord=2)
 hub_diff = np.linalg.norm(new_hub_scores- hub_scores, ord=2)

i=0
j=1
for i in range(len(authority)):
    for j in range(len(authority)):
        if(authority[i]>=authority[j]):
            out=authority[i];
            authority[i]=authority[j]
            authority[j]=out
        if(hub[i]>hub[j]):
            out=hub[i]
            hub[i]=hub[j]
            hub[j]=out
print("Ranking based on Hub Scores:")
for i in range(len(authority)):
    print("Rank" ,i+1,hub[i])
print("Ranking based on Authority Scores:")
for i in range(len(authority)):
    print("Rank",i+1,hub[i])


8.WebScraping On e-commerce platform using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')
products = soup.find_all('div', {'class': 'product-tuple-listing'})
for product in products:
    title = product.find('p', {'class': 'product-title'})
    price = product.find('span', {'class': 'product-price'})
    if price:
        product_price = convert_price_to_float(price.get('data-price', '0'))
    else:
        product_price = 0.0 
        rating = product.find('div', {'class': 'filled-stars'})
     if title and price:
        product_name = title.text.strip()
        product_rating = rating['style'].split(';')[0].split(':')[-1] if rating else "No rating"
        products_data.append({
        'Product': product_name,
        'Price': float(product_price),
        'Rating': product_rating
   })
   print(f'Product: {product_name}')
   print(f'Price: {product_price}')
   print(f'Rating: {product_rating}')
   print('---')
else:
  print('Failed to retrieve content')
  return products_data











