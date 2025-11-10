

# Interesting things found in the data

For an item like 

```
"Composto por dois empanados com frango, maionese, alface americana e pão com gergelim"  
translation: "Composed of two breaded chicken pieces, mayonnaise, iceberg lettuce, and a sesame seed bun"
```

the following JSON fields can be found in the data:

### `name`, `description`

Basic information.

### `images`

This contains an image URL with a presentation image for the item. We can parse the image with a multimodal LLM and extract more relevant keywords/information for the food item. 


### `vegan`

Useful for queries specifically asking for vegan or vegetarian food items.

### `search`

These keywords can be used as a shortcut to go from **query → food item**. If query contains or is very similar to these `search` field keywords, then we can bump up the score of the food item.

```json
{
    "count": 1,
    "method": "term",
    "term": "burguer king"
},
{
    "count": 1,
    "method": "suggestion",
    "term": "mcdonald's"
},
{
    "count": 1,
    "method": "suggestion",
    "term": "chillis hot dog"
},
{
    "count": 1,
    "method": "term",
    "term": "dog’s"
}

```

### `taxonomy` field

This can be used for tags. We can match up our query with certain tags and then bump up the score for food items with those tags.

```json
"taxonomy": {
    "l0": "ALIMENTOS_PREPARADOS",
    "l1": "SANDUICHES",
    "l2": "HAMBURGUERES"
},
```

### `category_name` field
This is similar to `taxonomy`, we just treat is as another tag.

# Search algorithms used in the solution

### vector search
We use embedding models to embed the data into high-dimensional vectors. We then build in-memory FAISS indexes from these vectors, so we can perform fast similarity search using inner product (dot product) distance metric, which gives us cosine similarity when the embeddings are normalized. 

### text (keyword) search

We BM25 for a fast and simple keyword search.

# Streamlined data format

This new data format is optimized for text search queries over the food items.

```json
"item_id": <the given itemId from the CSV>,  
"name": "The given item name",
"description": "The given description"
"combined_description": "A concise keyword-dense natural language description combining the given `description` field, the given `name` field and the extracted information from the image.",
"metrics": {
    "total_orders": <computed using the given total_orders field of the data>,
    "reorder_rate": <comnputed using the reorderRate field in the given data>,
}
"tags": [
    "burger",
    "vegetables",
    "fast food",
    "bread",
], 
"dietary_flags": {
    "vegan": false, // from the given `vegan` field
    "lactose_free": false, // from the given `lacFree` field
    "organic": false, // from the given organic field`
}
"associated_keyword_hooks": [
    "burger king",
    "sandwitch",
    "mcdonald's"
],
"co_purchased_items": [
    { 
        "item_id": "fd800d81-bbf5-4e75-81c0-ee91441c3333",
    }
]
```

For `tags` we use the terms provided in the `taxonomy` field, the string in the `category_name` field, as well as other LLM-generated keywords that classify the food item. 
These tags can be ingredient names (e.g chicken, maionese, tomatoes, etc), food classes, general food item names (eg. "pizza", "burger") etc.

`associated_keyword_hooks` is made from the `search` field. Only the terms are kept, the "method" and "count" are removed.

`co_purchased_items` is the trimmed down version of the `coPurchaseItems` field.

`reorder_rate` and `total_orders` fields will be used in ranking items, as we want to promote more popular and often re-ordered items.

### Future additions

##### Keep L0 L1 L2 on taxonomy
```json
do some hierarchical search ?
"taxonomy": {
    "l0": "ALIMENTOS_PREPARADOS",
    "l1": "SANDUICHES",
    "l2": "HAMBURGUERES"
},
```

##### Meal Timing Context
Rationale: Enables queries like "café da manhã" (breakfast) or "jantar" (dinner).

```json
"meal_context": {
    "primary_shifts": ["lunch", "dinner"],  // from orderingRate data
    "suitable_for": ["lunch", "dinner", "snack"]  // LLM-inferred
}
```

##### weigh the tags


# Search algorithm

The search pipeline (implemented in `match_query.py`) runs multiple parallel searches and intelligently combines their results through a multi-stage process:

## Pipeline Overview

### Stage 1: Parallel Multi-Index Search
Four different search methods run in parallel, each returning the top 30 items:

1. **BM25 Combined Description Index** - Keyword-based search on item names and descriptions
2. **FAISS Combined Description Index** - Semantic vector search on item descriptions using Voyage AI embeddings
3. **FAISS Tags/Hooks Index** - Semantic matching on tags and keyword hooks
   - Finds the top 30 most similar tags/hooks to the query
   - Scores items based on: (matched tags/hooks count) / (total tags/hooks for item)
   - Returns top 10 items with normalized scores (0-1 range)
4. **FAISS CLIP Image Index** - Visual-semantic matching on product images using CLIP embeddings
   - Encodes query text with CLIP multilingual model
   - Matches against CLIP-encoded product images

All scores are normalized to the 0-1 range for fair comparison.

### Stage 2: Score Aggregation
Scores from all four methods are aggregated with configurable multipliers:
- BM25 description scores: weight = 1.0
- FAISS description scores: weight = 1.0
- FAISS tags/hooks scores: weight = 0.5 (configurable via `bm25_tag_hooks_score_multiplier`)
- CLIP image scores: weight = 0.75 (configurable via `clip_image_index_score_multiplier`)

Items are ranked by their total aggregated score, and the top 50 candidates are shortlisted for reranking.

### Stage 3: AI Reranking
The shortlisted candidates are reranked using the Voyage AI rerank-2.5 model, which deeply understands semantic relevance between the query and item descriptions.

Two reranking strategies are available:
- **MULTIPLY** (default): `final_score = aggregated_score × (relevance_score ^ power)`
  - Combines search signals with AI relevance assessment
  - The `relevance_score_power` parameter (default 1.0) can emphasize the reranker's judgment
- **REPLACE**: `final_score = relevance_score`
  - Uses only the reranker's relevance score

### Stage 4: Final Ranking
Items are sorted by final score and returned. Future enhancements will include popularity boosts based on `total_orders` and `reorder_rate` metrics. 