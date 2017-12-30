FROM ubuntu:17.10
RUN apt-get update && apt-get install -y \
	python3-pip \
	python3-numpy \
	ffmpeg \
	python3.6 \
	xz-utils
RUN pip3 install -U \
	flask \
	flask_cors \
	librosa \
	google-api-python-client \
	pytube \
	matplotlib \
	pyfluidsynth \
	fluidsynth
WORKDIR /app
ADD . /app
EXPOSE 5000
CMD ["python3.6", "/app/Flask/url_to_chroma.py"]