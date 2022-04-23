from flask import Flask
import yact
import os
from flask import Flask, Config


# Build custom configuration object
class YactConfig(Config):
    """
    Customized config object
    Extends Flask config to support YAML via YACT
    """
    def from_yaml(self, config_file, directory=None):
        """
        All this method needs to do is load config
        from our config file, then *add* that config
        to the existing app config
        """
        config = yact.from_file(config_file, directory=directory)
        for section in config.sections:
            # Convention is to use all uppercase keys, we'll just force it
            self[section] = config[section]

# Override flask's default config class
Flask.config_class = YactConfig

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yml')
    app.config.from_yaml(config_path)

    # a simple page that says hello (for test)
    @app.route('/')
    def hello():
        return 'Hello! Welcome!'

    from lung import api
    app.register_blueprint(api.bp)

    return app
