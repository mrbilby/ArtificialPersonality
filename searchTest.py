
import requests

def google_custom_search(query, api_key, cx):
    """
    Perform a Google Custom Search query.

    Args:
        query (str): The search query.
        api_key (str): Your Google Custom Search API key.
        cx (str): The custom search engine ID.

    Returns:
        dict: The JSON response from the Google API, or None if an error occurs.
    """
    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": cx,
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an error for HTTP issues
        return response.json()  # Return the JSON response
    except requests.exceptions.RequestException as e:
        print(f"Error during Google Custom Search API call: {e}")
        return None

# Example usage
if __name__ == "__main__":
    search_query = "Python tutorials"
    api_key = 'AIzaSyCnzh-vxW31eYf43BoSTY9viOY-bTfvDpw' # Replace with your actual API key
    cx = "c1d5e6c731911430d"  # Replace with your actual cx ID
    
    results = google_custom_search(search_query, api_key, cx)
    if results:
        print("Search Results:")
        for item in results.get("items", []):
            print(f"Title: {item.get('title')}")
            print(f"Link: {item.get('link')}\n")

    """
    <script async src="https://cse.google.com/cse.js?cx=c1d5e6c731911430d">
</script>
<div class="gcse-search"></div>
    
    """