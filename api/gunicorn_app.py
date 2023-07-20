from gunicorn.app.base import BaseApplication
from flask_app import app

class GunicornApp(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()
        
    def load_config(self):
        for key, value in self.options.items():
            self.cfg.set(key, value)
        
    def load(self):
        return self.application
        
if __name__ == "__main__":
    options = {
        'bind': '0.0.0.0:5000',
        'workers': 1,
        'threads': 1,
        'accesslog': '-',
    }
    
    gunicorn_app = GunicornApp(app, options)
    gunicorn_app.run()
    