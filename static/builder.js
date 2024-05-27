var ndata = {
    "nodes":[
        {"id":"n1", "loaded":true, "style":{"label":"Node1", "fillColor":"rgba(236,46,46,0.8)"}, "x":0, "y":0, "locked": true},
        {"id":"n2", "loaded":true, "style":{"label":"Node2", "fillColor":"rgba(47,195,47,0.8)" }, "x":200, "y":50},
        {"id":"n3", "loaded":true, "style":{"labeL":"Node3", "fillColor":"rgba(28,124,213,0.8)" }, "x":100, "y":100},
        {"id":"n4", "loaded":true, "style":{"labeL":"Node4", "fillColor":"rgba(236,46,46,1)" }, "x":250, "y":250, "locked": true}
    ],
    "links":[
        {"id":"l1","from":"n1", "to":"n2"}
    ]
};
document.addEventListener("DOMContentLoaded", function () {


var t = new NetChart({
        container: "demo",
        area: { height: 350 },
        data: { preloaded: ndata },
        events:{
            onPointerDown: function(e, args) {
                updateInfo("pointer down");
                console.log("down", args);
            },
            onPointerUp: function(e, args) {
                updateInfo("pointer up");
                console.log("up", args);
                if(args.clickNode) {
                    var node = args.clickNode;
                    var onodes = getOverlappingNodes(node);
                    connectNodes(node, onodes);
                }
            },
            onPointerDrag: function(e, args) {
                updateInfo("dragging");
                console.log("drag", args);
            },
            onPointerMove: function(e, args) {
                //this is basically onMouseMove, but originally was named like this.
                //console.log("move", args);
            }
        }
    });
});


function getOverlappingNodes(node) {
    if(!node) return;

    var found = [];
    var dim = t.getNodeDimensions(node);

    var x = x1 = dim.x;
    var y = y1 = dim.y;
    var radius = dim.radius;

    //get all nodes:
    var nodes = t.nodes();
    for (var i = 0; i < nodes.length; i++) {
        var obj = nodes[i];
        //skip dragged node itself.
        if(obj.id === node.id) {
            continue;
        }
        var odim = t.getNodeDimensions(obj);
        var x0 = odim.x;
        var y0 = odim.y;

        var m = Math.sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0)) < radius;
        if(m) {
            found.push(obj);
        }
    }
    return found;
}

function connectNodes(node, onodes) {
    for (var i = 0; i < onodes.length; i++) {
        var onode = onodes[i];

        var link = {"id": "link-" + node.id + "-" + onode.id,"from": node.id, "to": onode.id, style: {"toDecoration": "arrow"}}
        t.addData({nodes:[],links: [link]});
    }
}

function updateInfo(info) {
    var element = document.getElementById("info");
    element.innerHTML = "Last event: " + info;
}
