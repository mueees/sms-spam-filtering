var convertData = require('../../lib/convert-data');

convertData.convertToJson(__dirname + '/spam-raw-data/SMSSpamCollection.txt', __dirname + '/spam-converted-data/data.json');
convertData.convertForOctave(__dirname + '/spam-raw-data/SMSSpamCollection.txt', __dirname + '/spam-converted-data/data.txt');
convertData.convertTxtToFeatureJson(__dirname + '/spam-raw-data/ex2data2.txt', __dirname + '/spam-converted-data/octaveFeature.json');