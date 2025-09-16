# gunicorn_config.py
# This tells Gunicorn how to run the app in production.
bind = "0.0.0.0:10000"
workers = 2
threads = 4
worker_class = "gthread"
timeout = 120
