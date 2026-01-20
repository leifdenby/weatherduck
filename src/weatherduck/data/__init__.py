from .dummy import (
    DummyWeatherDataset,
    TimeseriesDummyWeatherDataset,
    TimeseriesWeatherDataModule,
    WeatherDuckDataModule,
    build_dummy_weather_graph,
)

try:
    from .anemoi import AnemoiNativeGridDataModule
except ModuleNotFoundError:
    AnemoiNativeGridDataModule = None

__all__ = [
    "DummyWeatherDataset",
    "TimeseriesDummyWeatherDataset",
    "WeatherDuckDataModule",
    "TimeseriesWeatherDataModule",
    "build_dummy_weather_graph",
]
if AnemoiNativeGridDataModule is not None:
    __all__.append("AnemoiNativeGridDataModule")
