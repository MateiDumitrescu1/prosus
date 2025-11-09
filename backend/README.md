

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

### `category_name` field
This is similar to `taxonomy`, we just treat is as another tag.


### `taxonomy` field

This can be used for tags. We can match up our query with certain tags and then bump up the score for food items with those tags.

```json
"taxonomy": {
    "l0": "ALIMENTOS_PREPARADOS",
    "l1": "SANDUICHES",
    "l2": "HAMBURGUERES"
},
```

# Search algorithms used in the solution
### vector search
### text search
BM25

# New streamlined data format

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


# Search strategy 

Use multiple search avenues and combine the scores / results.

### tag search
**query** ---(semantic distance)---> **related tags** ---(we hit those tags and for each item, sum up the points for each tageted tag. if tags are not weighted, all count as 1 point )---> **ranked list of items based on total tag score**

As of right now tags are not weighted, this will be implemented later. If tags are not weighted, we are basically ranking the items based on how many of their tags were considered relevant to the search query.

### semantic search on `description`

### text search on `description`
