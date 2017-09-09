# Music

## Instructions:
Install Docker
```
docker build -t music .
docker run -p 5000:5000 music
```
localhost:5000

## To do:
1. ~~Extract audio from Youtube~~
1. Use Librosa to extract chromagram
1. UI Design
1. UI Implementation
1. Create a training set of notes with added noise

## Basic Functionality:
* Input Youtube URL
* Display Chromagram moving along with the music as it is playing

## Possible extension:
1. Transcribe sheet music

# Required Packages
* librosa
* google-api-python-client
* pytube
* matplotlib
