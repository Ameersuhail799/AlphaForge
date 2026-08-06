from config.settings import DEFAULT_SYMBOL, START_DATE

from src.data.providers.yahoo_provider import YahooProvider
from src.data.validator import DataValidator


provider = YahooProvider()

validator = DataValidator()

df = provider.download(
    DEFAULT_SYMBOL,
    START_DATE,
)

report = validator.validate(df)

print()

print(report)