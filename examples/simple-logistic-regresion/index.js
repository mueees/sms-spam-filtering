var LogisticRegression = require('../../lib/mind').LogisticRegression;

var logisticRegression = new LogisticRegression({

});

logisticRegression.addTrainingData({
    input: {
        color: 1,
        fontSize: 14,
        distance: 60
    },
    output: 1
});
logisticRegression.addTrainingData({
    input: {
        color: 1,
        distance: 50,
        fontSize: 12
    },
    output: 1
});
logisticRegression.addTrainingData({
    input: {
        fontSize: 16,
        color: 2,
        distance: 80
    },
    output: 1
});

logisticRegression.addTrainingData({
    input: {
        color: 3,
        fontSize: 20,
        distance: 20
    },
    output: 0
});
logisticRegression.addTrainingData({
    input: {
        color: 3,
        fontSize: 22,
        distance: 30
    },
    output: 0
});
logisticRegression.addTrainingData({
    input: {
        fontSize: 26,
        color: 2,
        distance: 15
    },
    output: 0
});

logisticRegression.addTrainingData({
    input: {
        fontSize: 22,
        color: 3,
        distance: 25
    },
    output: 0
});

logisticRegression.on('train:after:iteration', function (data) {
    console.log(data.previousCost);
    console.log(data.currentCost);
    console.log('/--------------------------/');
});
logisticRegression.on('train:after', function (data) {
    console.log('Finish iterations: ' + data.iteration);
});

logisticRegression.train();

console.log('Prediction: ' + logisticRegression.predict({
    color: 3,
    fontSize: 22,
    distance: 30
}));