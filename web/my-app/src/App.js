import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';
//import d3 from 'd3';
var d3 = require('d3'); // Why does this work but not import?
d3.scale = require('d3-scale');

class App extends Component {
  render() {
    return (
      <div className="App">
        <div className="App-header">
          <img src={logo} className="App-logo" alt="logo" />
          <h2>Welcome to React</h2>
        </div>
        <p className="App-intro">
          To get started, edit <code>src/App.js</code> and save to reload.
        </p>
				<SearchBar/>
				<ChromaGraph/>
      </div>
    );
  }
}

class SearchBar extends Component {
  render() {
		return (<div>
			<input type="text"/><input type="submit" value="Search"/>
		</div>);
  }
}

class SoundWave extends Component {
	constructor(props) {
		super(props);
		this.state = {
			data: null
		};
		this.fetchData();
	}
	fetchData() {
		var that = this;
		//fetch("http://localhost:5000/dummydata")
		var hostname = window.location.hostname;
		var url = "http://"+hostname+":5000/dl/FM7MFYoylVs"
		fetch(url)
			.then(function(response){
				window.x = response;
				return response.json();
			}).then(function(data){
				that.setState({data: data});
			}).catch(function(error) {
				  console.log('There has been a problem with your fetch operation: ' + error.message);
			});
	}
	getData() {
		if (this.state.data != null) {
			return this.state.data;
		}
		var x = [];
		for (var i = 0; i < 100; i++) {
			x.push(Math.random());
		}
		return x;
	}
	render() {
		return (<svg width='500' height='200' ref='svg'></svg>);
	}
	drawGraph() {
		var data = this.getData();
		var node = this.refs.svg;
		var svg = d3.select(node);
		svg.selectAll("*").remove();
		var g = svg.append("g");

		var svgRect = svg.node().getBoundingClientRect();
		var xScale = d3.scale.scaleLinear()
			.domain([0, data.length])
			.range([0,svgRect.width]);
		var yScale = d3.scale.scaleLinear()
			.domain(d3.extent(data))
			.range([0,svgRect.height]);
		var scaledPoints = [];
		for (var i = 0; i < data.length; i++) {
			scaledPoints.push([xScale(i), yScale(data[i])]);
		}
		var lineGenerator = d3.line();
		var pathData = lineGenerator(scaledPoints);
		var path = g.append('path');
		path.attr('d', pathData)
			.attr('fill', 'none')
			.attr('stroke', '#000');
	}
	componentDidMount() {
		this.drawGraph();
	}
	componentDidUpdate() {
		this.drawGraph();
	}
}

class ChromaGraph extends Component {
	constructor(props) {
		super(props);
		this.state = {
			data: null
		};
		this.fetchData();
	}
	fetchData() {
		var that = this;
		var hostname = window.location.hostname;
		var url = "http://"+hostname+":5000/chroma/FM7MFYoylVs"
		fetch(url)
			.then(function(response){
				window.x = response;
				return response.json();
			}).then(function(data){
				that.setState({data: data});
			}).catch(function(error) {
				  console.log('There has been a problem with your fetch operation: ' + error.message);
			});
	}
	getData() {
		if (this.state.data != null) {
			return this.state.data;
		}
		var x = [];
		for (var j = 0; j < 12; j++) {
			var y = []
			for (var i = 0; i < 100; i++) {
				y.push(Math.random());
			}
			x.push(y);
		}
		return x;
	}
	render() {
		return (<svg width='700' height='200' ref='svg'></svg>);
	}
	drawGraph() {
		var data = this.getData();
		var node = this.refs.svg;
		var svg = d3.select(node);
		svg.selectAll("*").remove();
		var g = svg.append("g");

		var svgRect = svg.node().getBoundingClientRect();
		var xScale = d3.scale.scaleLinear()
			.domain([0, data.length])
			.range([0,svgRect.width]);
		var yScale = d3.scale.scaleLinear()
			.domain([0,1])
			.range([0,svgRect.height/12.5]);
		var scaledPoints = data.map(function(d) {
					return d.map(function(p, i) {
						return [xScale(i), yScale(p)];
					});
				});
		var lineGenerator = d3.line();
		var dist = svgRect.height/12.0;
		scaledPoints.forEach(function(sp,i) {
			var pathData = lineGenerator(sp);
			var path = g.append('path');
			path.attr('d', pathData)
				.attr('fill', 'none')
				.attr('stroke', '#000')
				.attr('transform', 'translate(0,'+(dist*i)+')');
		});
	}
	componentDidMount() {
		this.drawGraph();
	}
	componentDidUpdate() {
		this.drawGraph();
	}
}

export default App;
