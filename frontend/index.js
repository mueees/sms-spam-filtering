var mind = require('../lib/mind'),
    _ = require('lodash'),
    rawDocuments = require('./data/data.js');

var lambda = 0.2,
    alpha = 0.05,
    maxIterations = 750;

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
    /*var input = {
     documentLength: document.text.length,
     wordsCount: wordsCount(document.text),
     commaCount: commaCount(document.text),
     exclamationCount: exclamationCount(document.text),
     capitalLetterLength: countUpperCaseChars(document.text),
     isContainsLongNumber: Number(containsLongNumber(document.text))
     };*/

    return {
        input: [1, countUpperCaseChars(document.text), containsLongNumber(document.text)],
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

    var docs = _.cloneDeep(documents);

    _.each(docs, function (document) {
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

    return docs;
}

// convert raw documents to features
var featureDocuments = transformToFeature(rawDocuments);

var spamDocs = _.filter(featureDocuments, function (d) {
    return d.output
});
var notSpamDocs = _.filter(featureDocuments, function (d) {
    return !d.output
});

var resultDocuments = spamDocs.concat(_.slice(notSpamDocs, 0, spamDocs.length + 100));

var documents = _.shuffle(resultDocuments);

// var documents = mapFeature(rawDocuments, 6);
var countTrainingDocuments = Math.round(documents.length * 0.75);

var trainingDocuments = _.slice(documents, 0, countTrainingDocuments);
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
    costVsIterationData.push(data);
});

_.each(trainingDocuments, function (trainingDocument) {
    logisticRegression.addTrainingData(trainingDocument);
});

logisticRegression.train();

/*Highcharts.chart('boundary', {
 chart: {},
 title: {
 text: 'Raw Data'
 },
 xAxis: {
 title: {
 enabled: true,
 text: 'Feature 1'
 }
 },
 yAxis: {
 title: {
 text: 'Feature 2'
 }
 },
 legend: {
 layout: 'vertical',
 align: 'left',
 verticalAlign: 'top',
 x: 100,
 y: 70,
 floating: true,
 backgroundColor: (Highcharts.theme && Highcharts.theme.legendBackgroundColor) || '#FFFFFF',
 borderWidth: 1
 },
 plotOptions: {
 scatter: {
 marker: {
 radius: 5,
 states: {
 hover: {
 enabled: true,
 lineColor: 'rgb(100,100,100)'
 }
 }
 },
 states: {
 hover: {
 marker: {
 enabled: false
 }
 }
 },
 tooltip: {
 headerFormat: '<b>{series.name}</b><br>',
 pointFormat: '{point.x} cm, {point.y} kg'
 }
 }
 },
 series: [
 {
 type: 'scatter',
 name: 'Positive',
 color: 'rgba(223, 83, 83, .5)',
 data: _(rawDocuments).filter(function (document) {
 return Boolean(document.output)
 }).map(function (document) {
 return document.input
 }).value()

 },
 {
 type: 'scatter',
 name: 'Negative',
 color: 'rgba(119, 152, 191, .5)',
 data: _(rawDocuments).filter(function (document) {
 return !Boolean(document.output)
 }).map(function (document) {
 return document.input
 }).value()
 }
 ]
 });*/

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

/**
 *
 * Calculate Acuracy
 * */

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

/*var learningCurveData = [];

for (var i = 10; i < trainingDocuments.length; i = i + 300) {
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
});*/

var btn = document.getElementById('answer');
var input = document.getElementById('question');

btn.addEventListener('click', function (event) {
    event.preventDefault();

    var feature = transformDocumentToFeature({
        text: input.value
    });

    var prediction = logisticRegression.predict(feature.input);

    console.log(prediction);
});