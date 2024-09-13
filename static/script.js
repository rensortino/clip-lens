let highlightTraces = [];
let originalColors = []


function setupHeatmapHovering(heatmapElement) {
    heatmapElement.on('plotly_hover', function(data) {
        const point = data.points[0];
        const reductionType = document.getElementById('reduction').value
        const scatterId = "scatter_" + (reductionType == "pca" ? "pca" : "umap")
        Plotly.Fx.hover(scatterId, [
            {pointNumber: point.x},
            {pointNumber: point.y}
        ])
        updateImages(point.x, point.y)
    });
    heatmapElement.on('plotly_unhover', function(data) {
        document.getElementById('image1').src = "";
        document.getElementById('image2').src = "";
    });
}

function setupScatterHovering(scatterElement) {
    scatterElement.on('plotly_hover', function(data) {
        const point = data.points[0];
        const index = point.pointNumber;
        Plotly.Fx.hover("heatmap", [
            {pointNumber: [index, index]}
        ])
        updateImages(index, index)
    });
    scatterElement.on('plotly_unhover', function(data) {
        document.getElementById('image1').src = "";
        document.getElementById('image2').src = "";
    });
}

function updateImages(row, col) {
    fetch(`/get_image_pair/${row}/${col}`)
        .then(response => response.json())
        .then(json => {
            document.getElementById('image1').src = json.image1;
            document.getElementById('image2').src = json.image2;
        });
}