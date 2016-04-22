var numeric = require('numeric'),
    _ = require('lodash'),
    util = require('util'),
    EventEmitter = require('events').EventEmitter;

function LogisticRegression(options) {
    options = options || {};

    var defaultOptions = {
        costThreshold: 0.0005
    };

    this.options = _.assign(defaultOptions, _.pick(options, ['costThreshold']));

    /**
     * X = [
     *      doc1.feature1  doc1.feature2  doc1.feature3
     *      doc2.feature1  doc2.feature2  doc2.feature3
     *      doc3.feature1  doc3.feature2  doc3.feature3
     *      ]
     * */
    this.X = []; // features matrix

    /**
     * y = [
     *      doc1.result1  doc1.result2
     *      doc2.result1  doc2.result2
     *      doc3.result1  doc3.result2
     *      ]
     * */
    this.y = []; // results matrix

    this.alpha = options.alpha || 0.001;

    this.lambda = options.lambda || 0.1;

    this.maxIterations = options.maxIterations || 5000;

    this.theta = [];
}

util.inherits(LogisticRegression, EventEmitter);

_.extend(LogisticRegression.prototype, {
    /**
     * @description
     * Our hypothesis function
     *
     * @param {Number} z
     * @returns {Number}
     * */
    sigmoid: function (z) {
        return numeric.div(1, numeric.add(1, numeric.exp(numeric.mul(-1, z))))
    },

    /**
     * @description
     * Calculate cost value for current Theta
     * */
    cost: function (X, y, theta, lambda) {
        /*
         * costSum = (hypothesis - y);
         * regularization = lambda * (sum(theta^2))
         * cost = (1/2*m) * (costSum^2 + regularization);
         * */

        var m = X.length; // training examples size

        var costSum = this.costSum(X, y, theta);

        costSum = numeric.mul(costSum, costSum);

        var regularization = lambda * numeric.sum(numeric.mul(theta, theta));

        return (1 / 2 * m) * (numeric.sum(costSum) + regularization);
    },

    gradientDescent: function (theta, lambda, alpha, X, y) {
        var gradient = this.gradient(theta, lambda, X, y);

        // numeric.sub Pointwise x-y
        // numeric.mul Pointwise x*y
        return numeric.sub(theta, numeric.mul(alpha, gradient));
    },

    costSum: function (X, y, theta) {
        // (hypothesis - y);

        var z = numeric.dot(X, theta); // vector

        var hypothesis = this.sigmoid(z); // vector

        return numeric.sub(hypothesis, y)
    },

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
    gradient: function (theta, lambda, X, y) {
        /*
         * costSum = (hypothesis - y);
         * gradientSum = X'*costSum;
         * regularization = (lambda/m)*theta;
         * gradient = (1/m)*gradientSum + regularization ;
         * */

        var m = X.length;  // data row length

        var regularization = numeric.mul(lambda, theta); // vector

        regularization = numeric.div(regularization, m); // vector

        var costSum = this.costSum(X, y, theta);

        var gradientSum = numeric.dot(numeric.transpose(X), costSum); // vector

        return numeric.add(regularization, numeric.div(gradientSum, m));
    },

    /**
     * @description
     * Initialize theta vector
     * */
    initializeTheta: function () {
        // vector [ [0], [0], ... ]
        this.theta = numeric.rep([this.X[0].length, 1], 0);
    },

    /**
     * @addTrainingData
     * Add new data for training model
     *
     * @param {Vector} feature
     * @param {Vector} result
     * */
    addTrainingData: function (data) {
        this.X.push(this.convertInputToVector(data.input));
        this.y.push([data.output]);
    },

    /**
     * @description
     * Convert user input to feature vector
     *
     * @param {Object} input
     * @returns {Array} Converted vector feature
     * */
    convertInputToVector: function (input) {
        var vector = [],
            me = this;

        // this.featureSchema - ['keyName1', 'keyName2']
        if (!this.featureSchema) {
            this.featureSchema = [];

            _.forOwn(input, function (value, key) {
                me.featureSchema.push(key);
            });
        }

        _.each(this.featureSchema, function (keyName) {
            vector.push(input[keyName]);
        });

        return vector;
    },

    /**
     * @description
     * Train model
     * */
    train: function () {
        this.initializeTheta();

        var i = 0;

        if (this.X.length) {
            while (i < this.maxIterations) {
                this.emit('train:before:iteration', {
                    iteration: i,
                    theta: this.theta,
                    currentCost: this.getCost()
                });

                this.theta = this.gradientDescent(this.theta, this.lambda, this.alpha, this.X, this.y);

                this.emit('train:after:iteration', {
                    iteration: i,
                    theta: this.theta,
                    currentCost: this.getCost()
                });

                i++;
            }

            this.emit('train:after', {
                iteration: i,
                theta: this.theta,
                currentCost: this.getCost()
            });
        }
    },

    /**
     * @description
     * Predict value, based on training model
     * */
    predict: function (feature) {
        var thetaT = numeric.transpose(this.theta);

        var x = numeric.transpose([this.convertInputToVector(feature)]);

        return this.sigmoid(numeric.dot(thetaT, x))[0][0];
    },

    getCost: function () {
        return this.cost(this.X, this.y, this.theta, this.lambda);
    },

    resetData: function () {
        this.X = [];
        this.y = [];
    }
});

module.exports = LogisticRegression;