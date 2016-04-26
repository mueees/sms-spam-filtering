var mind = require('../lib/mind'),
    _ = require('lodash'),
    rawDocuments = require('./data/octaveFeature.js');

var lambda = 1,
    alpha = 0.004,
    maxIterations = 2300;

function countUpperCaseChars(str) {
    var count = 0, len = str.length;
    for (var i = 0; i < len; i++) {
        if (/[A-Z]/.test(str.charAt(i))) count++;
    }
    return count;
}

function wordsCount(str) {
    return str.split(' ').length;
}

function commaCount(str) {
    return str.split(',').length - 1;
}

function exclamationCount(str) {
    return str.split('!').length - 1;
}

function containsLongNumber(str) {
    return /[0-9]{3,}/.test(str);
}

function transformToFeature(documents) {
    return _.map(documents, transformDocumentToFeature);
}

function transformDocumentToFeature(document) {
    var input = {
        documentLength: document.text.length,
        wordsCount: wordsCount(document.text),
        commaCount: commaCount(document.text),
        exclamationCount: exclamationCount(document.text),
        capitalLetterLength: countUpperCaseChars(document.text),
        isContainsLongNumber: Number(containsLongNumber(document.text))
    };

    return {
        input: input,
        raw: document,
        output: Number(document.spam)
    }
}

function mapFeature(documents, degree) {
    /*
     * degree = 6;
     x0
     x1 + x2 +
     x1^2 + x1x2 + x2^2 +
     x1^3 + x2^3 + x1^2x2 + x1*x2^2x
     x1^4 + x1^3x2 + x1^2x2^2 + x1x2^3 + x2^4
     x1^5 + x1^4x2 + x1^3x2^2 + x1^2x2^3 + x1x2^4 + x2^5
     x1^6 + x1^5x2 + x1^4x2^2 + x1^3x2^3 + x1^2x2^4 + x1x2^5 + x2^6
     * */

    _.each(documents, function (document) {
        var input = document.input;
        var x1 = input[0];
        var x2 = input[1];

        var resultInput = [];

        for (var i = 1; i <= degree; i++) {
            for (var k = 0; k <= i; k++) {
                resultInput.push(Math.pow(x1, (i - k)) * Math.pow(x2, k));
            }
        }

        resultInput.unshift(1);

        document.input = resultInput;
    });
}

// convert raw documents to features
// var documents = transformToFeature(rawDocuments);
mapFeature(rawDocuments, 6);
var documents = rawDocuments;
var countTrainingDocuments = Math.round(documents.length * 0.7);

var trainingDocuments = _.slice(documents, 0/*, countTrainingDocuments*/);
var testDocuments = _.slice(documents, countTrainingDocuments);

// for cost-vs-iteration chart
var costVsIterationData = [];

var logisticRegression = new mind.LogisticRegression({
    alpha: alpha,
    lambda: lambda,
    maxIterations: maxIterations,
    costThreshold: false
});

logisticRegression.on('train:after:iteration', function (data) {
    /*console.log('Iteration: ' + data.iteration);
     console.log('Cost Value: ' + data.currentCost);
     console.log('/----------------------/');*/

    costVsIterationData.push(data);
});


_.each(trainingDocuments, function (trainingDocument) {
    logisticRegression.addTrainingData(trainingDocument);
});

console.log(logisticRegression.getCost());
console.log(logisticRegression.getGradient());

logisticRegression.train();

console.log(logisticRegression.getCost());

console.log(_.map(logisticRegression.theta, function (thetaValue) {
    return thetaValue[0];
}));

Highcharts.chart('cost-vs-iteration', {
    title: {
        text: 'Cost Value vs Learning Iteration',
        x: -20 //center
    },
    xAxis: {
        categories: _.map(costVsIterationData, function (row) {
            return row.iteration;
        })
    },
    yAxis: {
        title: {
            text: 'Cost Value'
        },
        plotLines: [{
            value: 0,
            width: 1,
            color: '#808080'
        }]
    },
    legend: {
        layout: 'vertical',
        align: 'right',
        verticalAlign: 'middle',
        borderWidth: 0
    },
    series: [{
        name: 'Cost Value',
        data: _.map(costVsIterationData, function (row) {
            return row.currentCost;
        })
    }]
});

/*
function getAccuracy(testDocuments, model) {
    var correctDocuments = [],
        falseDocuments = [];

    _.each(testDocuments, function (testDocument) {
        var prediction = Math.round(model.predict(testDocument.input));

        if (prediction === Number(testDocument.output)) {
            correctDocuments.push(testDocument);
        } else {
            falseDocuments.push(testDocument);
        }
    });

    return {
        accuracy: correctDocuments.length / testDocuments.length,
        correctDocuments: correctDocuments,
        falseDocuments: falseDocuments
    };
}

var accuracy = getAccuracy(testDocuments, logisticRegression);
var rightSpam = _.filter(accuracy.correctDocuments, function (doc) {
    return doc.raw.spam;
});

console.log('Accuracy: ' + accuracy.accuracy);
console.log('Right Spam Answer: ' + rightSpam.length);

// LEARNING CURVE

var learningCurveData = [];

for (var i = 10; i < trainingDocuments.length; i = i + 1000) {
    var stepData = {
        m: i
    };

    var trainingLearningCurveDocuments = _.slice(trainingDocuments, 0, i);
    var learningCurveLG = new mind.LogisticRegression({
        alpha: 0.001,
        maxIterations: maxIterations,
        lambda: lambda,
        costThreshold: false
    });

    _.each(trainingLearningCurveDocuments, function (trainingDocument) {
        learningCurveLG.addTrainingData(trainingDocument);
    });

    learningCurveLG.train();
    stepData.traningCost = learningCurveLG.getCost();

    learningCurveLG.resetData();
    _.each(testDocuments, function (trainingDocument) {
        learningCurveLG.addTrainingData(trainingDocument);
    });

    stepData.testCost = learningCurveLG.getCost();

    learningCurveData.push(stepData);

    if (i % 10 === 0) {
        console.log(i);
    }
}

Highcharts.chart('learning-curve', {
    title: {
        text: 'Cost Value vs Learning Iteration',
        x: -20 //center
    },
    xAxis: {
        categories: _.map(learningCurveData, function (row) {
            return row.m;
        })
    },
    yAxis: {
        title: {
            text: 'Cost Value'
        },
        plotLines: [{
            value: 0,
            width: 1,
            color: '#808080'
        }]
    },
    legend: {
        layout: 'vertical',
        align: 'right',
        verticalAlign: 'middle',
        borderWidth: 0
    },
    series: [
        {
            name: 'Training Cost Value',
            data: _.map(learningCurveData, function (row) {
                return row.traningCost;
            })
        },
        {
            name: 'Test Cost Value',
            data: _.map(learningCurveData, function (row) {
                return row.testCost;
            })
        }
    ]
});
*/
