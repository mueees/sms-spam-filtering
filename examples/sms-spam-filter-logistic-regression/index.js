/*
 *
 * 1. Convert data to features
 *
 * */

var _ = require('lodash'),
    natural = require('natural');

// prepare raw data: convert txt file -> json
require('./prepare-data');

var textData = require('./spam-converted-data/data.json');

var X = [],
    y = [];

_.each(textData, function (dataRow) {
    var tokens = natural.PorterStemmer.tokenizeAndStem(dataRow.text);
});