from pydantic_settings import BaseSettings, YamlConfigSettingsSource


class BaseYAMLSettings(BaseSettings):
    @classmethod
    def from_yaml(cls, path):
        class _ConfiguredSettings(cls):
            @classmethod
            def settings_customise_sources(
                cls,
                settings_cls,
                init_settings,
                env_settings,
                dotenv_settings,
                file_secret_settings,
            ):
                return (
                    init_settings, 
                    dotenv_settings, 
                    YamlConfigSettingsSource(settings_cls, path)
                )
        return _ConfiguredSettings()
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings, 
            dotenv_settings, 
            YamlConfigSettingsSource(settings_cls)
        )
