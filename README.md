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
  * Using [pretty_midi](https://github.com/craffel/pretty-midi) library, create random synthesized notes (varying velocity, instrument, notes)

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

# Approach

* Train a classifier which can determine which notes are played.
  * From a chromagram
  * From a sound wave (Use an RNN?)
* Write a function which can take the list of notes played, and determine what the potential chords and root note is
  * Use [music21](http://web.mit.edu/music21/doc/index.html)?
  * [PyChoReLib](http://chordrecognizer.sourceforge.net/)
* Find key of music
  * Can be provided by librosa?
  * Train classifier
  * [Using interval profile](http://www.cp.jku.at/research/papers/Madsen_Widmer_icmc_2007.pdf)
  * [Using pitch profile](http://rnhart.net/articles/key-finding/)
  * Other readings: [A Bayesian key-finding model](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.332.3337&rep=rep1&type=pdf)
* Given the music's key, we can determine the most likely chords.
  * Hooktheory has an API which gives probabilities for each chord sequence. Their dataset seems to be mainly composed of western music, so we'd have to find a way of doing this for music from other cultures. This can start off as a simple markovian transition model.
* Combine the chord probabilities based on the previous chord and the sound waves.

# Datasets

* [Nsynth](https://magenta.tensorflow.org/datasets/nsynth)
* [https://www.audiocontentanalysis.org/data-sets/](https://www.audiocontentanalysis.org/data-sets/)
