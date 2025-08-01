import wikipedia
import os

wikipedia.set_lang("en")

articles = {
    "Customer service": "customer_service.txt",
    "Billing (telecommunications)": "billing.txt",
    "Internet access": "internet_access.txt"
}

output_dir = "data/knowledge_base"
os.makedirs(output_dir, exist_ok=True)

for title, filename in articles.items():
    try:
        print(f"Downloading: {title}")
        content = wikipedia.page(title).content

        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)

        print(f"✓ Saved to {filename}")
    except Exception as e:
        print(f"✗ Error downloading '{title}': {e}")
