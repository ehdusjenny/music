FROM ubuntu:17.10
WORKDIR /app
ADD . /app
RUN apt-get update && apt-get install -y \
	python3-pip \
	python3-numpy \
	ffmpeg \
	xz-utils \
	python3.6
RUN pip3 install -U \
	flask \
	librosa \
	google-api-python-client \
	pytube \
	matplotlib
EXPOSE 5000
CMD ["python3.6", "/app/Flask/url_to_chroma.py"]