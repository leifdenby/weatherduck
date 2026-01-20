from .dummy import (
    DummyWeatherDataset,
    TimeseriesDummyWeatherDataset,
    TimeseriesWeatherDataModule,
    WeatherDuckDataModule,
    build_dummy_weather_graph,
)

try:
    from .anemoi import AnemoiWeatherDataModule
except ModuleNotFoundError:
    AnemoiWeatherDataModule = None

__all__ = [
    "DummyWeatherDataset",
    "TimeseriesDummyWeatherDataset",
    "WeatherDuckDataModule",
    "TimeseriesWeatherDataModule",
    "build_dummy_weather_graph",
]
if AnemoiWeatherDataModule is not None:
    __all__.append("AnemoiWeatherDataModule")
