var convertData = require('./lib/convert-data');

convertData.convertToJson(__dirname + '/spam-raw-data/SMSSpamCollection.txt', __dirname + '/spam-converted-data/data.json');