#!/bin/bash
gunicorn src.api:app -b 0.0.0.0:$PORT --workers 1 --threads 1