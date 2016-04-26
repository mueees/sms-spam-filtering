var lineReader = require('line-reader'),
    _ = require('lodash'),
    fs = require('fs');

function countUpperCaseChars(str) {
    var count = 0, len = str.length;
    for (var i = 0; i < len; i++) {
        if (/[A-Z]/.test(str.charAt(i))) count++;
    }
    return count;
}

exports.convertToJson = function (filePath, outFile) {
    var data = [];

    lineReader.eachLine(filePath, function (line, last, cb) {
        if (line.indexOf('ham') === 0) {
            data.push({
                text: line.slice(3).trim(),
                spam: false
            });
        } else {
            data.push({
                text: line.slice(4).trim(),
                spam: true
            });
        }

        if (last) {
            var shuffledData = _.shuffle(data);

            fs.writeFile(outFile, JSON.stringify(shuffledData), function (err) {
                if (err) {
                    console.log(err)
                }
            });
        } else {
            cb();
        }
    });
};

exports.convertForOctave = function (filePath, outFile) {
    var data = [];

    lineReader.eachLine(filePath, function (line, last, cb) {
        var text,
            output;

        if (line.indexOf('ham') === 0) {
            output = 0;
            text = line.slice(3).trim();
        } else {
            output = 1;
            text = line.slice(4).trim();
        }

        var feature1 = text.length;             // document length
        var feature2 = text.split(' ').length;  // words count
        var feature3 = text.split(',').length - 1;  // count ,
        var feature4 = Number(/[0-9]{3,}/.test(text));  // contains long numbers
        var feature5 = countUpperCaseChars(text);  // count capital letters

        data.push(feature5 + ',' + feature4 + ',' + output);

        if (last) {
            var shuffledData = _.shuffle(data);
            var result = shuffledData.join('\n');

            fs.writeFile(outFile, result, function (err) {
                if (err) {
                    console.log(err)
                }
            });
        } else {
            cb();
        }
    });
};

exports.convertTxtToFeatureJson = function (filePath, outFile) {
    var data = [];

    lineReader.eachLine(filePath, function (line, last, cb) {
        var parsedData = line.split(',');
        var output = parsedData.splice(parsedData.length - 1);

        for(var i = 0; i < parsedData.length; i++){
            parsedData[i] = Number(parsedData[i]);
        }

        data.push({
            input: parsedData,
            output: Number(output[0]),
            raw: line
        });

        if (last) {
            // var shuffledData = _.shuffle(data);

            fs.writeFile(outFile, JSON.stringify(data), function (err) {
                if (err) {
                    console.log(err)
                }
            });
        } else {
            cb();
        }
    });
};