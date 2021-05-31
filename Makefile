help:
	@echo 'make install:          Install package, and database'

install: 
	python scr/dispersion/scripts/setup_dispersion.py
	pip install -r requirements.txt
	pip install -e .

test:
	pytest
