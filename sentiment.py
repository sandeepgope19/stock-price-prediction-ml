from textblob import TextBlob
from newsapi import NewsApiClient

newsapi = NewsApiClient(api_key="ff8ab10bab574bd59808d2f5875cfaba")

def get_news_sentiment(query):
    articles = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=10)

    polarity_list = []
    headlines = []

    for article in articles['articles']:
        title = article['title']
        headlines.append(title)
        analysis = TextBlob(title)
        polarity_list.append(analysis.sentiment.polarity)

    if len(polarity_list) == 0:
        return 0, []

    avg_sentiment = sum(polarity_list) / len(polarity_list)

    return avg_sentiment, headlines