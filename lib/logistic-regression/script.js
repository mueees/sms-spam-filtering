var Mue = Mue || {};

Mue.LogisticRegression = function () {
    this.X = []; // features
    this.y = []; // results

    this.alpha = 0.1;
    this.lambda = 0;

    // vector [ [0], [0], ... ]
    this.theta = numeric.rep([Mue.LogisticRegression.extendFeatures(0, 0).length, 1], 0);

    // initialize canvas
    this.initCanvas();

    // start main loop
    this.loop();
};

/**
 * @description
 * Our hypothesis function
 *
 * @param {Number} z
 * @returns {Number}
 * */
Mue.LogisticRegression.sigmoid = function (z) {
    return numeric.div(1, numeric.add(1, numeric.exp(numeric.mul(-1, z))))
};

/**
 * @description
 * Gradient for calculation Gradient descent
 *
 * @param {Vector} theta
 * @param {Number} lambda
 * @param {Matrix} X
 * @param {Vector} y
 *
 * @returns {Vector}
 * */
Mue.LogisticRegression.gradient = function (theta, lambda, X, y) {
    /*
     * gradientSum = X'*(hypothesis - y);
     * grad = (1/m)*gradientSum + lambda*theta/m;
     * */

    var m = X.length;  // data row length

    var regression = numeric.mul(lambda, theta); // vector

    regression = numeric.div(regression, m); // vector

    var z = numeric.dot(X, theta); // vector

    var hypothesis = Mue.LogisticRegression.sigmoid(z); // vector

    var gradientSum = numeric.dot(numeric.transpose(X), numeric.sub(hypothesis, y)); // vector

    return numeric.add(regression, numeric.div(gradientSum, m));
};

/**
 * @description
 * Gradient descent
 *
 * @param {Vector} theta
 * @param {Number} lambda
 * @param {Number} alpha
 * @param {Matrix} X
 * @param {Vector} y
 *
 * @returns {Vector} Returns theta vector
 * */
Mue.LogisticRegression.gradientDescent = function (theta, lambda, alpha, X, y) {
    var gradient = Mue.LogisticRegression.gradient(theta, lambda, X, y);

    // numeric.sub: Pointwise x-y
    return numeric.sub(theta, numeric.mul(alpha, gradient));
};

Mue.LogisticRegression.extendFeatures = function (x1, x2) {
    return [
        1,
        x1,
        x2,
        x1 * x1,
        x2 * x2,
        x1 * x2
    ];
};

Mue.LogisticRegression.prototype.addDataRow = function (feature, result) {
    var extendedFeature = Mue.LogisticRegression.extendFeatures(feature[0], feature[1]);

    this.X.push(extendedFeature);
    this.y.push([result]);
};

Mue.LogisticRegression.prototype.initCanvas = function () {
    var me = this,
        canvas = document.getElementById('canvas'),
        lambda = document.getElementById('lambda'),
        groupInput = document.getElementById('group');

    // canvas handler
    this.canvas = new Processing(canvas);
    this.canvas.size(400, 400);

    canvas.onclick = function (evt) {
        if (typeof evt.offsetX == 'undefined') {
            evt.offsetX = evt.layerX - canvas.offsetLeft;
        }
        if (typeof evt.offsetY == 'undefined') {
            evt.offsetY = evt.layerY - canvas.offsetTop;
        }

        var a = (evt.offsetX / 200) - 1;
        var b = (evt.offsetY / 200) - 1;

        me.addDataRow([a, b], me.group);
    };

    // group handler
    this.group = getGroup();
    groupInput.onchange = function () {
        me.group = getGroup();
    };
    function getGroup() {
        return parseFloat(groupInput.value);
    }

    // lambda handler
    lambda.value = this.lambda;
    lambda.onchange = function () {
        me.lambda = lambda.value;
    }
};

Mue.LogisticRegression.prototype.render = function () {
    var thetaT = numeric.transpose(this.theta);

    this.canvas.scale(10);
    // Render background
    for (var i = 0; i < 40; i++) {
        for (var j = 0; j < 40; j++) {
            var a = j / 20 - 1;
            var b = i / 20 - 1;

            var x = numeric.transpose([Mue.LogisticRegression.extendFeatures(a, b)]);

            var value = Mue.LogisticRegression.sigmoid(numeric.dot(thetaT, x)[0][0]);

            this.canvas.stroke(255 * value, 100, 255 * (1 - value));

            this.canvas.point(j, i);
        }
    }

    this.canvas.scale(0.1);
    this.canvas.stroke(0);

    // Render points
    for (i = 0; i < this.X.length; i++) {
        if (this.y[i][0] > 0.5) {
            this.canvas.fill(255, 100, 50);
        } else {
            this.canvas.fill(50, 100, 255);
        }

        this.canvas.ellipse((this.X[i][1] + 1) * 200, (this.X[i][2] + 1) * 200, 5, 5);
    }
};

// main loop
Mue.LogisticRegression.prototype.loop = function () {
    var me = this;

    if (this.X.length) {
        for (var i = 0; i < 100; i++) {
            this.theta = Mue.LogisticRegression.gradientDescent(this.theta, this.lambda, this.alpha, this.X, this.y);
        }
    }

    this.render();

    setTimeout(function () {
        me.loop();
    }, 5);
};