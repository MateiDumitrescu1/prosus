test_query = "How do neural networks learn from data?"

# 20 sentences with varying relevance to the query
test_documents = [
    # Highly relevant (about neural networks and learning)
    "Neural networks learn by adjusting weights through backpropagation algorithms.",
    "Deep learning models update their parameters using gradient descent optimization.",
    "Training a neural network involves feeding it labeled examples and minimizing loss.",
    "Backpropagation calculates gradients to update network weights during training.",

    # Moderately relevant (about ML/AI in general)
    "Machine learning algorithms can identify patterns in large datasets.",
    "Artificial intelligence systems improve their performance over time.",
    "Supervised learning requires labeled training data for model development.",
    "Data preprocessing is crucial for building accurate predictive models.",

    # Somewhat relevant (tangentially related to tech/data)
    "Python is a popular programming language for data science.",
    "Big data technologies enable processing of massive information streams.",
    "Cloud computing provides scalable infrastructure for AI applications.",
    "Database systems store and organize structured information efficiently.",

    # Not relevant (completely different topics)
    "The Mediterranean climate is characterized by hot, dry summers.",
    "Soccer is the most popular sport worldwide with billions of fans.",
    "Pasta carbonara is a classic Italian dish made with eggs and cheese.",
    "The Great Wall of China stretches over 13,000 miles.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "Jazz music originated in New Orleans in the early 20th century.",
    "Mount Everest is the tallest mountain on Earth at 29,032 feet.",
    "The periodic table organizes chemical elements by atomic structure."
]

sentence_transformers_clip_image__model_name = "clip-ViT-B-32"
sentence_transformers_clip_multilingual_text__model_name = "clip-ViT-B-32-multilingual-v1"