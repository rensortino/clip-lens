// let heatmapPlot, scatterPlot;
const highlightColor = "#FFFF00"
let highlightTraces = [];
let originalColors = []


function setupHeatmapHovering(heatmapElement) {
    heatmapElement.on('plotly_hover', function(data) {
        const point = data.points[0];
        // const index = point.pointNumber;
        // highlightHeatmapRowCol(point)
        const reductionType = document.getElementById('reduction').value
        const scatterId = "scatter_" + (reductionType == "pca" ? "pca" : "umap")
        Plotly.Fx.hover(scatterId, [
            {pointNumber: point.x},
            {pointNumber: point.y}
        ])
        updateImages(point.x, point.y)
    });
    heatmapElement.on('plotly_unhover', function(data) {
        unHighlightHeatmap();
        document.getElementById('image-heatmap1').src = "";
        document.getElementById('image-heatmap2').src = "";
        // unHighlightScatterPoint(data)
    });
}

function setupScatterHovering(scatterElement) {
    scatterElement.on('plotly_hover', function(data) {
        const point = data.points[0];
        const index = point.pointNumber;
        Plotly.Fx.hover("heatmap", [
            {pointNumber: index},
        ])
        // originalColors = [...data.points[0].data.marker.color]
        // highlightScatterPoint(data)
        updateImage(index)
    });
    scatterElement.on('plotly_unhover', function(data) {
        // unHighlightScatterPoint(data);
        document.getElementById('image-scatter').src = "";
        // unHighlightHeatmapCell()
    });
}


function setupCrossHovering() {

    heatmapElement.on('plotly_hover', function(data) {
        const point = data.points[0];
        // const index = point.pointNumber;
        // highlightHeatmapRowCol(point)
        updateImages(point.x, point.y)
    });

    scatterElement.on('plotly_hover', function(data) {
        const point = data.points[0];
        const index = point.pointNumber;
        highlightScatterPoint(data)
        updateImage(index)
    });

    heatmapElement.on('plotly_unhover', function(data) {
        unHighlightHeatmap();
        document.getElementById('image-heatmap1').src = "";
        document.getElementById('image-heatmap2').src = "";
        // unHighlightScatterPoint(data)
    });
    scatterElement.on('plotly_unhover', function(data) {
        unHighlightScatterPoint(data);
        document.getElementById('image-heatmap1').src = "";
        document.getElementById('image-heatmap2').src = "";
        // unHighlightHeatmapCell()
    });
}

function highlightScatterPoint(data) {
    var pn='',
        tn='',
        colors=[],
        sizes=[];

    for(var i=0; i < data.points.length; i++){
        pn = data.points[i].pointNumber;
        tn = data.points[i].curveNumber;
        colors = data.points[i].data.marker.color;
        sizes = data.points[i].data.marker.size;
    };
    colors[pn] = '#C54C82';
    sizes[pn] = 16;

    var update = {'marker':{color: colors, size:sizes}};
    
    // Ugly but working solution
    const plotId = data.event.target.parentElement.parentElement.parentElement.parentElement.parentElement.parentElement.id
    Plotly.restyle(plotId, update, [tn]);
}

function highlightHeatmapRowCol(points) {
    const rowIndex = points.y;
    const colIndex = points.x;
    const cols = points.data.z.length;
    const rows = points.data.z[0].length;

    const heatmapElement = document.getElementById('heatmap');
    
    // Create a new trace for the highlighted row
    const highlightedRow = {
        z: [Array(cols).fill(1)],  // Use 1 to represent the maximum value
        x: Array.from({length: cols}, (_, i) => i),
        y: [rowIndex],
        type: 'heatmap',
        showscale: false,
        colorscale: [[0, 'rgba(255,0,0,0)'], [1, 'rgba(255,0,0,0.5)']],  // Red with 50% opacity
        hoverinfo: 'none'
      };
      
      // Create a new trace for the highlighted column
      const highlightedCol = {
        z: Array(rows).fill([1]),  // Use 1 to represent the maximum value
        x: [colIndex],
        y: Array.from({length: rows}, (_, i) => i),
        type: 'heatmap',
        showscale: false,
        colorscale: [[0, 'rgba(255,0,0,0)'], [1, 'rgba(255,0,0,0.5)']],  // Red with 50% opacity
        hoverinfo: 'none'
      };

    // Remove existing highlight traces if any
    unHighlightHeatmap();

    // Add new highlight traces
    Plotly.addTraces('heatmap', [highlightedRow, highlightedCol]).then(() => {
        highlightTraces = [heatmapElement.data.length - 2, heatmapElement.data.length - 1];
      });
    
}


function unHighlightHeatmap() {
    if (highlightTraces.length > 0) {
        Plotly.deleteTraces('heatmap', highlightTraces);
        highlightTraces = [];
      }
}

function unHighlightScatterPoint(data) {
    var pn='',
        tn='',
        colors=[],
        sizes=[];
    for(var i=0; i < data.points.length; i++){
        pn = data.points[i].pointNumber;
        tn = data.points[i].curveNumber;
        colors = data.points[i].data.marker.color;
        sizes = data.points[i].data.marker.size
    };
    colors[pn] = originalColors[pn];
    sizes[pn] = 10;

    var update = {'marker':{color: colors, size:sizes}};

    // Ugly but working solution
    const plotId = data.event.target.parentElement.parentElement.parentElement.parentElement.parentElement.parentElement.id
    Plotly.restyle(plotId, update, [tn]);

}

function updateImage(index) {
    fetch(`/get_image/${index}`)
        .then(response => response.json())
        .then(json => {
            document.getElementById('image-scatter').src = json.image;
        });
}

function updateImages(row, col) {
    fetch(`/get_image_pair/${row}/${col}`)
        .then(response => response.json())
        .then(json => {
            document.getElementById('image-heatmap1').src = json.image1;
            document.getElementById('image-heatmap2').src = json.image2;
        });
}