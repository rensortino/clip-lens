let highlightColor = 'green';
let originalColors = []
let scatterSelected = -1;


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
    if (originalColors.length === 0){
        originalColors = [...scatterElement.data[0].marker.color];
    }

    scatterElement.on('plotly_hover', function(data) {
        const point = data.points[0];
        const index = point.pointNumber;
        if (scatterSelected === -1) {
            Plotly.Fx.hover("heatmap", [
                {pointNumber: [index, index]}
            ])
            updateImages(index, index)
        } else {
            Plotly.Fx.hover("heatmap", [
                {pointNumber: [scatterSelected, index]}
            ])
            updateImages(scatterSelected, index)
        }
    });

    scatterElement.on('plotly_click', function(data) {
        const point = data.points[0];
        const index = point.pointNumber;

        var pn='',
            tn='';
        for(var i=0; i < data.points.length; i++){
            pn = data.points[i].pointNumber;
            tn = data.points[i].curveNumber;
            colors = data.points[i].data.marker.color;
        };

        if (scatterSelected !== -1)
            colors[scatterSelected] = originalColors[scatterSelected]

        if (index === scatterSelected) {
            scatterSelected = -1
            colors[pn] = originalColors[pn];
        } else {
            scatterSelected = index
            colors[pn] = highlightColor;
        }

        var update = {'marker':{color: colors, size: 12}};
        Plotly.restyle(scatterElement.id, update, [tn]);
    });

    scatterElement.on('plotly_unhover', function(data) {
        if (scatterSelected === -1) {
            document.getElementById('image1').src = "";
            document.getElementById('image2').src = "";
        } else {
            document.getElementById('image2').src = "";
        }
    });
}

function updateImages(row, col) {
    fetch(`/get_image_pair/${row}/${col}`)
        .then(response => response.json())
        .then(json => {
            document.getElementById('image1').src = json.image1;
            document.getElementById('image2').src = json.image2;
            document.getElementById('caption1').innerText = json.caption1;
            document.getElementById('caption2').innerText = json.caption2;
        });
}