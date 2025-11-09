import { useState } from 'react'
import type { FormEvent } from 'react'

interface SearchResult {
  query: string
  item_ids: string[]
  count: number
}

interface ItemDetails {
  itemId: string
  name: string
  description: string
  price: number
  images: string[]
  category_name: string
}

const API_BASE = 'http://localhost:8080'

function App() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult | null>(null)
  const [items, setItems] = useState<ItemDetails[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [rerankStrategy, setRerankStrategy] = useState<'multiply' | 'replace'>('replace')

  const handleSearch = async (e: FormEvent) => {
    e.preventDefault()

    if (!query.trim()) {
      setError('Please enter a search query')
      return
    }

    setIsLoading(true)
    setError(null)
    setItems([])

    try {
      // Get search results (item IDs)
      const searchResponse = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim(),
          top_k: 30,
          rerank_strategy: rerankStrategy
        })
      })

      if (!searchResponse.ok) {
        throw new Error(`Search failed: ${searchResponse.statusText}`)
      }

      const searchData: SearchResult = await searchResponse.json()
      setResults(searchData)

      // Fetch details for each item
      const itemDetailsPromises = searchData.item_ids.map(async (itemId) => {
        const response = await fetch(`${API_BASE}/items/${itemId}`)
        if (response.ok) {
          return response.json()
        }
        return null
      })

      const itemsData = await Promise.all(itemDetailsPromises)
      const validItems = itemsData.filter((item): item is ItemDetails => item !== null)
      setItems(validItems)

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred while searching')
      setResults(null)
      setItems([])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <h1 className="text-4xl font-bold text-gray-900 mb-8 text-center">
          Hybrid Search
        </h1>

        {/* Search Form */}
        <form onSubmit={handleSearch} className="mb-8">
          <div className="flex gap-2 mb-3">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search for food items..."
              className="flex-1 px-4 py-3 text-lg border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading}
              className="px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {isLoading ? 'Searching...' : 'Search'}
            </button>
          </div>

          {/* Strategy Selector */}
          <div className="flex items-center gap-3">
            <label className="text-sm font-medium text-gray-700">
              Rerank Strategy:
            </label>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setRerankStrategy('replace')}
                disabled={isLoading}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                  rerankStrategy === 'replace'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                Replace
              </button>
              <button
                type="button"
                onClick={() => setRerankStrategy('multiply')}
                disabled={isLoading}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                  rerankStrategy === 'multiply'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                Multiply
              </button>
            </div>
          </div>
        </form>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-800 font-medium">Error: {error}</p>
          </div>
        )}

        {/* Results */}
        {results && (
          <div>
            <div className="mb-6">
              <h2 className="text-2xl font-semibold text-gray-900">
                Results for "{results.query}"
              </h2>
              <p className="text-sm text-gray-600 mt-1">
                Found {results.count} {results.count === 1 ? 'item' : 'items'}
              </p>
            </div>

            {/* Item Cards Grid */}
            {items.length > 0 ? (
              <div className="grid gap-4">
                {items.map((item) => (
                  <div
                    key={item.itemId}
                    className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow"
                  >
                    <div className="flex gap-4 p-4">
                      {/* Image */}
                      <div className="flex-shrink-0">
                        {item.images && item.images.length > 0 ? (
                          <img
                            src={`${API_BASE}/images/${item.images[0]}`}
                            alt={item.name}
                            className="w-24 h-24 object-cover rounded-md"
                            onError={(e) => {
                              e.currentTarget.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96"%3E%3Crect fill="%23f3f4f6" width="96" height="96"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" fill="%239ca3af" font-size="14"%3ENo Image%3C/text%3E%3C/svg%3E'
                            }}
                          />
                        ) : (
                          <div className="w-24 h-24 bg-gray-200 rounded-md flex items-center justify-center">
                            <span className="text-xs text-gray-500">No Image</span>
                          </div>
                        )}
                      </div>

                      {/* Content */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between gap-2 mb-1">
                          <h3 className="text-lg font-semibold text-gray-900 truncate">
                            {item.name}
                          </h3>
                          <span className="text-xs text-gray-400 font-mono flex-shrink-0">
                            {item.itemId}
                          </span>
                        </div>
                        {item.description && (
                          <p className="text-sm text-gray-600 mt-1">
                            {item.description}
                          </p>
                        )}
                        <div className="flex items-center gap-3 mt-2">
                          <span className="text-lg font-bold text-green-600">
                            ${item.price.toFixed(2)}
                          </span>
                          {item.category_name && (
                            <span className="text-xs px-2 py-1 bg-gray-100 text-gray-700 rounded">
                              {item.category_name}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 text-center py-8">
                No results found
              </p>
            )}
          </div>
        )}

        {/* Initial State */}
        {!results && !error && !isLoading && (
          <div className="text-center text-gray-500 mt-12">
            <p className="text-lg">Enter a search query to find food items</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
