import React, { Component } from 'react';
import { Orchestra } from 'react-orchestra/web';

const midiURL = 'https://s3-eu-west-1.amazonaws.com/ut-music-player/assets/midis/beet1track-medium-fast.mid';
export class MidiPlayer extends Component {
    constructor(props) {
        super(props);
        this.state = {
            playSong: false,
        };
        this.onPlayToggle = this.onPlayToggle.bind(this);
    }
    onPlayToggle() {
        console.log("Inside onPlayToggle");
        this.setState({playSong: !this.state.playSong});
    }
    render() {
        return (
            <Orchestra
                midiURL={midiURL}
                play={this.state.playSong}
                selectedTracks={[0, 1]}
                loader={OrchestraLoader}>
                <button onClick={this.onPlayToggle} className="btn btn-primary">{this.state.playSong ? "Stop" : "Play"} Music</button>
            </Orchestra>
        );
    }
}

class OrchestraLoader extends Component {
    render() {
        return (
            <div>Loading...</div>
        )
    }
}