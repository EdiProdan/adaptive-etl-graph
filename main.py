from src.phase1_extraction import WikipediaClient


if __name__ == "__main__":
    client = WikipediaClient()

    popular_titles = client.get_popular_pages(limit=10000)  # Start small for testing

    # Collect their data
    pages = client.collect_pages(popular_titles, "data/input/pages/base_pages.json")

    print(f"Successfully collected {len(pages)} Wikipedia pages!")
    print(f"Sample page: {pages[0].title if pages else 'None'}")
