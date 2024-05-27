// // note that the chart container has `ondragover` and `ondrop` event handlers registered
// // and the draggable items have `onselectstart`, `ondragstart` and `ondragend`.
//
// /** The node on which the mouse cursor is currently hovering. */
// var currentHoverNode = null;
// var draggedColor = null;
// var nodeColorMap = {};
//
// function colorSelectStart(event, elem) {
//     // workaround to enable drag-n-drop in IE9: http://stackoverflow.com/q/5500615/1711598
//     if (elem.dragDrop) {
//         elem.dragDrop();
//     }
//
//     return false;
// }
//
// function colorDragStart(event, elem) {
//     // cannot use `event.dataTransfer.setData`: http://stackoverflow.com/a/11959389/1711598
//     draggedColor = elem.style.backgroundColor;
//
//     // but it is needed for Firefox otherwise draggin does not start.
//     event.dataTransfer.setData("text", draggedColor);
// }
//
// function colorDragEnd(event) {
//     draggedColor = null;
// }
//
// function networkDragOver(event) {
//     if (draggedColor && currentHoverNode) {
//         // this instructs the browser to allow the drop
//         event.preventDefault();
//     }
// }
//
// function networkDragDrop(event) {
//     if (draggedColor && currentHoverNode) {
//         nodeColorMap[currentHoverNode.id] = draggedColor;
//         t.updateStyle();
//     }
//
//     event.preventDefault();
// }
//
// function networkHoverChanged(event) {
//     currentHoverNode = event.hoverNode;
// }
//
// function networkNodeStyleFunction(node) {
//     var color = nodeColorMap[node.id];
//     if (color)
//         node.fillColor = color;
// }
//
// function buildData() {
//
// }
//
//
// document.addEventListener("DOMContentLoaded", function () {
//     var t = new NetChart({
//         container: "demo",
//         area: {height: window.screen.height - 150},
//         data: {url: "http://127.0.0.1:5000/get_nodes_n_links"},
//         nodeMenu: {enabled: false},
//         linkMenu: {enabled: false},
//         style: {
//             nodeAutoScaling: "linear",
//             nodeDetailMinSize: 0
//         },
//         layout: {
//             nodeSpacing: 20
//         },
//         navigation: {
//             mode: "showall"
//         },
//
//         theme: NetChart.themes.dark,
//
//         interaction: {
//             resizing: {enabled: false},
//             zooming: {
//                 zoomExtent: [0.1, 2],
//                 autoZoomExtent: [0.1, 1]
//             }
//         }
//
//     });
// });

document.addEventListener("DOMContentLoaded", function () {
    var randomSeed = 10; // this can be changed to generate different data sets
    var nextNodeId = 0;
    var iter = 80;
    var chart = null;


    function fetchCurrentTime() {
        fetch('http://127.0.0.1:5000/get_current_time')
            .then(response => response.json())
            .then(data => {
                const timeElement = document.getElementById('currentTime');
                const pathElement = document.getElementById('paths_traversed');
                if (timeElement) {
                    timeElement.textContent = data.time; // Set the fetched time
                }
                if (pathElement) {
                    pathElement.textContent = data.paths_traversed.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ","); // Set the fetched time
                }
            })
            .catch(error => console.error('Error fetching current time:', error));
    }

    function buildData(nodeList, success, fail) {
        // build a random graph
        var links = [];
        var nodes = [];

        fetch('http://127.0.0.1:5000/get_nodes_n_links')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json(); // Parse the response body as JSON
            })
            .then(data => {
                nodes = data.nodes;
                links = data.links;


                success({"nodes": nodes, "links": links}); // Call success function with data
            })
            .catch(error => {
                console.error('Error:', error); // Handle any errors that occur during the request
                fail(); // Call fail function on error
            });
    }

    function linkStyle(link) {
        link.length = 1;
        link.radius = 10;
        // link.fromDecoration = "circle";
        link.toDecoration = "hollow arrow";

        var highlighted_or_not = link.data.is_highlighted || 0; // Default to 0 if undefined

        // Dynamic color scale based on 'is_highlighted'
        var startColor = {r: 255, g: 255, b: 255}; // White
        var endColor = {r: 255, g: 0, b: 0}; // Red
        var colorRatio = highlighted_or_not > 1000 ? 1 : highlighted_or_not / 1000; // Cap at 10 for full intensity

        var interpolatedColor = {
            r: Math.round(startColor.r + (endColor.r - startColor.r) * colorRatio),
            g: Math.round(startColor.g + (endColor.g - startColor.g) * colorRatio),
            b: Math.round(startColor.b + (endColor.b - startColor.b) * colorRatio),
        };

        link.fillColor = `rgb(${interpolatedColor.r},${interpolatedColor.g},${interpolatedColor.b})`;

        if (highlighted_or_not > 0) {
            link.lineDash = 1;
            link.radius = 15;
        } else {
            link.fillColor = "#FFFFFF"; // Default to white if not highlighted
        }

        link.items = [
            {   // Default item places just as the regular label.
                text: link.data.weight,
                padding: 2,
                backgroundStyle: {
                    fillColor: "rgba(0,0,0, 1)",
                },
                textStyle: {
                    fillColor: "white",
                    font: "15px FontAwesome"
                }
            },
        ];
    }

    chart = new NetChart({
        container: document.getElementById("demo"),
        data: {dataFunction: buildData},
        area: {height: window.screen.height - 250},

        nodeMenu: {enabled: false},
        linkMenu: {enabled: false},

        style: {
            node: {
                display: "roundtext",
            },
            nodeLabel: {
                align: "center",
                textStyle: {
                    fillColor: "#000000",
                    font: "15px FontAwesome"
                }
            },

            nodeAutoScaling: "linear",
            nodeDetailMinSize: 0,
            linkStyleFunction: linkStyle,
        },

        credits: {
            image: ""
        },

        layout: {
            mode: "hierarchy",
            nodeSpacing: 300,
            gravity: {
                from: "auto",
                to: "nearestLockedNode",
                strength: 0.001,
            },
        },
        navigation: {
            mode: "showall"
        },

        theme: NetChart.themes.dark,
        legend: {enabled: true},

        title: {
            text: "Data (traffic) Architecture Simulator"
        },

        toolbar: {},

        selection: {},

        interaction: {
            resizing: {enabled: true},
            nodeMenu: {showData: true},
            selection: {linksSelectable: false},
            rotation: {fingers: true},
            zooming: {
                zoomExtent: [0.1, 9],
                autoZoomExtent: [0.1, 1]
            }
        }
    });

    // Reload data every 5 seconds (5000 milliseconds)
    var intervalHandle = setInterval(function () {
        chart.reloadData();
    }, 75);

    fetchCurrentTime(); // Fetch and display the current time immediately on load
    // Optionally, update the time at fixed intervals (e.g., every 10 seconds)
    setInterval(fetchCurrentTime, 75);

    function disposeDemo() {
        clearInterval(intervalHandle); // Clear the interval
        disposeDemo = null; // Clear the disposeDemo function
        intervalHandle = null; // Clear the interval handle
    }
});





