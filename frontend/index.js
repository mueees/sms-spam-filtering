var mind = require('../lib/mind'),
    _ = require('lodash'),
    rawDocuments = require('./data/data.js');

var lambda = 0,
    maxIterations = 500;

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

    input.a = input.documentLength * input.documentLength;
    input.b = input.wordsCount * input.wordsCount;
    input.c = input.commaCount * input.commaCount;
    input.d = input.exclamationCount * input.exclamationCount;
    input.e = input.capitalLetterLength * input.capitalLetterLength;
    input.g = input.isContainsLongNumber * input.isContainsLongNumber;

    /*input.two_capitalLetterLength = input.capitalLetterLength * input.capitalLetterLength;
     input.three_capitalLetterLength = input.capitalLetterLength + input.capitalLetterLength;

     input.a = input.isContainsLongNumber * input.capitalLetterLength;
     input.b = input.isContainsLongNumber + input.capitalLetterLength;*/

    return {
        input: input,
        raw: document,
        output: Number(document.spam)
    }
}

// convert raw documents to features
var documents = transformToFeature(rawDocuments);
var countTrainingDocuments = Math.round(documents.length * 0.7);

var trainingDocuments = _.slice(documents, 0, countTrainingDocuments);
var testDocuments = _.slice(documents, countTrainingDocuments);

// for cost-vs-iteration chart
var costVsIterationData = [];

var logisticRegression = new mind.LogisticRegression({
    alpha: 0.001,
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

logisticRegression.train();

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