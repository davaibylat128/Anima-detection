from flask import Flask

def create_app():
    app = Flask(__name__)
    # Register blueprints or initialize extensions here
    return app

app = create_app()
