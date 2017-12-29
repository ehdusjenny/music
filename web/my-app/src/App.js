import React, { Component } from 'react';
import logo from './logo.svg';
import './css/app.css';
import {SearchBar} from './components/SearchBar'
import {MidiPlayer} from './components/MidiPlayer'

export class App extends Component {
  render() {
    return (
      <div className="app">
        <div className="app-header">
          <img src={logo} className="app-logo" alt="logo" />
          <h2>Welcome to React</h2>
        </div>
        <SearchBar/>
        <MidiPlayer/>
      </div>
    );
  }
}

export default App;
