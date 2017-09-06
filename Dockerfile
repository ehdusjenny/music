FROM python:3.6.2
WORKDIR /app
ADD . /app
RUN wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-64bit-static.tar.xz
RUN tar xvf ffmpeg-git-*.tar.xz
RUN mv /app/ffmpeg-git-20170904-64bit-static/ffmpeg /app/Flask
RUN rm -rf /app/ffmpeg-git-*
RUN pip install numpy
RUN pip install flask librosa google-api-python-client pytube matplotlib
RUN echo 'alias ffmpeg="/app/Flask/ffmpeg"' >> ~/.bashrc
EXPOSE 5000
CMD ["python", "/app/Flask/url_to_chroma.py"]