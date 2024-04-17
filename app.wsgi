#!/usr/bin/python
import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/html/bidBoost-RAG-api")

activate_this = '/var/www/html/bidBoost-RAG-api/.venv/bin/activate'
with open(activate_this) as file_:
	exec(file_.read(), dict(__file__=activate_this))

from app import app as application
application.secret_key = 'something super SUPER secret'
