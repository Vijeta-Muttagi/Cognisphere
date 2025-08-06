# 🔧 Install Gradio (if not installed)
!pip install gradio --quiet
!pip install transformers selenium beautifulsoup4 pandas --quiet
!apt-get update > /dev/null
!apt install chromium-chromedriver --quiet
# 📦 Imports
import time
import pandas as pd
import gradio as gr
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from transformers import pipeline

# 🧠 Get headless Chrome driver
def get_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    return webdriver.Chrome(options=options)

# 🤖 Review Agent
class ReviewAgent:
    def __init__(self):
        self.pipeline = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

    def scrape_amazon(self, url, max_reviews=20):
        try:
            driver = get_driver()
            driver.get(url)
            time.sleep(3)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            driver.quit()
            review_elements = soup.select('.review-text-content span')
            reviews = [el.get_text(strip=True) for el in review_elements[:max_reviews] if el]
            return reviews
        except Exception as e:
            return [f"Error scraping Amazon: {e}"]

    def scrape_flipkart(self, url, max_reviews=20):
        try:
            driver = get_driver()
            driver.get(url)
            time.sleep(3)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            driver.quit()
            review_elements = soup.select('div.t-ZTKy')
            reviews = [el.get_text(strip=True).replace("READ MORE", "") for el in review_elements[:max_reviews]]
            return reviews
        except Exception as e:
            return [f"Error scraping Flipkart: {e}"]

    def detect_fakes(self, reviews):
        try:
            results = self.pipeline(reviews)
            return list(zip(reviews, results))
        except Exception as e:
            return [("Error", {"label": "error", "score": 0.0})]

    def summarize(self, results, platform):
        df = pd.DataFrame(results, columns=["Review", "Prediction"])
        df['Stars'] = df['Prediction'].apply(lambda x: int(x['label'][0]) if isinstance(x, dict) else 0)
        df['Confidence'] = df['Prediction'].apply(lambda x: round(x['score'], 2) if isinstance(x, dict) else 0.0)
        df['Label'] = df['Stars'].apply(lambda x: 'FAKE' if x <= 2 else 'REAL')

        fake_count = (df['Label'] == 'FAKE').sum()
        real_count = (df['Label'] == 'REAL').sum()
        total = len(df)
        fake_ratio = round((fake_count / total) * 100, 2) if total else 0.0

        summary = (
            f"📊 **{platform} Summary**\n\n"
            f"- Total Reviews: {total}\n"
            f"- FAKE Reviews: {fake_count} ({fake_ratio}%)\n"
            f"- REAL Reviews: {real_count} ({100 - fake_ratio}%)\n"
        )

        if fake_ratio > 50:
            summary += "\n🚨 **WARNING:** High number of suspicious reviews!"
        elif total < 5:
            summary += "\n⚠️ **Caution:** Not enough data for a confident result."
        else:
            summary += "\n✅ **Looks Trustworthy** based on available data."

        return summary
# 🎛️ Gradio frontend
agent = ReviewAgent()

def analyze_links(amazon_url, flipkart_url):
    output = []

    if amazon_url.strip():
        amazon_reviews = agent.scrape_amazon(amazon_url)
        if len(amazon_reviews) > 0:
            amazon_results = agent.detect_fakes(amazon_reviews)
            amazon_summary = agent.summarize(amazon_results, "Amazon")
            output.append(amazon_summary)
        else:
            output.append("❌ No reviews found on Amazon.")

    if flipkart_url.strip():
        flipkart_reviews = agent.scrape_flipkart(flipkart_url)
        if len(flipkart_reviews) > 0:
            flipkart_results = agent.detect_fakes(flipkart_reviews)
            flipkart_summary = agent.summarize(flipkart_results, "Flipkart")
            output.append(flipkart_summary)
        else:
            output.append("❌ No reviews found on Flipkart.")

    return "\n\n---\n\n".join(output)

# 🚀 Launch Gradio UI
gr.Interface(
    fn=analyze_links,
    inputs=[
        gr.Textbox(label="🔗 Amazon Product URL", placeholder="https://amzn.in/d/a1Jynsi"),
        gr.Textbox(label="🔗 Flipkart Product URL", placeholder="https://dl.flipkart.com/s/NXCPZwuuuN"),
    ],
    outputs=gr.Markdown(label="📊 Review Summary"),
    title="🕵️ Fake Review Detector for Amazon & Flipkart",
    description="Paste product links to detect fake reviews using AI.",
).launch()