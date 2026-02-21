import yfinance as yf

def load_stock_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)