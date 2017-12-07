import React, { Component } from 'react';
import * as d3 from 'd3';

export class SoundWave extends Component {
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
            console.log(data);
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
        return [x];
    }
    render() {
        return (<svg width='500' height='200' ref='svg'></svg>);
    }
    drawGraph() {
        var data = this.getData()[0];
        var node = this.refs.svg;
        var svg = d3.select(node);
        svg.selectAll("*").remove();
        var g = svg.append("g");

        var svgRect = svg.node().getBoundingClientRect();
        var xScale = d3.scaleLinear()
            .domain([0, data.length])
            .range([0,svgRect.width]);
        var yScale = d3.scaleLinear()
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