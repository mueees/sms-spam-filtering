var LogisticRegressionApparatus = require('apparatus/lib/apparatus/classifier/logistic_regression_classifier'),
    LogisticRegression = require('../../lib/mind').LogisticRegression,
    _ = require('lodash'),
    rawDocuments = require('./data.json');

var logisticRegression = new LogisticRegression({
    alpha: 0.001,
    maxIterations: 400,
    costThreshold: false
});
var logisticRegressionApparatus = new LogisticRegressionApparatus();

function countUpperCaseChars(str) {
    var count = 0, len = str.length;
    for (var i = 0; i < len; i++) {
        if (/[A-Z]/.test(str.charAt(i))) count++;
    }
    return count;
}

function countWords(str) {
    return str.split(' ').length;
}

function transformToFeature(documents) {
    return _.map(documents, function (document) {
        var input = {
            documentLength: document.text.length,
            wordsCount: countWords(document.text),
            capitalLetterLength: countUpperCaseChars(document.text),

            /*words*/
            callW: (document.text.match(/call/gi) || []).length,
            buy: (document.text.match(/buy/gi) || []).length,
            urgentW: (document.text.match(/urgent/gi) || []).length
        };

        return {
            input: input,
            raw: document,
            output: Number(document.spam)
        }
    });
}

function transformToApparatusFeature(document) {
    var inputForApparatus = [];

    _.forOwn(document.input, function (value, key) {
        inputForApparatus.push(value);
    });

    return inputForApparatus;
}

var shuffledDocuments = _.slice(_.shuffle(rawDocuments), 0, 300);
//var shuffledDocuments = _.slice(rawDocuments, 0, 50);

// convert raw documents to features
var documents = transformToFeature(shuffledDocuments);
var countTrainingDocuments = Math.round(documents.length * 0.8);
var trainingDocuments = _.slice(documents, 0, countTrainingDocuments);
var testDocuments = _.slice(documents, countTrainingDocuments);

_.each(trainingDocuments, function (trainingDocument) {
    logisticRegression.addTrainingData(trainingDocument);
    logisticRegressionApparatus.addExample(transformToApparatusFeature(trainingDocument), trainingDocument.output);
});

logisticRegression.train();
logisticRegressionApparatus.train();

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

function getApparatusAccuracy(testDocuments, model) {
    var correctDocuments = [],
        falseDocuments = [];

    _.each(testDocuments, function (testDocument) {
        var prediction = Math.round(model.classify(transformToApparatusFeature(testDocument.input)));

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

var mindAccuracy = getAccuracy(testDocuments, logisticRegression);
var apparatusAccuracy = getApparatusAccuracy(testDocuments, logisticRegressionApparatus);

console.log('/----------------------------/');
console.log('Mind accuracy: ' + mindAccuracy.accuracy);
console.log('Apparatus accuracy: ' + apparatusAccuracy.accuracy);

var mindRightSpam = _.filter(mindAccuracy.correctDocuments, function (doc) {
    return doc.raw.spam;
});
var apparatusRightSpam = _.filter(apparatusAccuracy.correctDocuments, function (doc) {
    return doc.raw.spam;
});

console.log('/----------------------------/');
console.log('Mind: Right spam: ' + mindRightSpam.length);
console.log('ApparatusRightSpam: Right spam: ' + apparatusRightSpam.length);
console.log('/----------------------------/');