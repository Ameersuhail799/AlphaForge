from config.settings import DEFAULT_SYMBOL, START_DATE

from src.data.providers.yahoo_provider import YahooProvider


provider = YahooProvider()

df = provider.download(
    symbol=DEFAULT_SYMBOL,
    start_date=START_DATE,
)

print(df.head())

print()

print(df.tail())

print()

print(df.shape)