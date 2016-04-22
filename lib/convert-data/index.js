var lineReader = require('line-reader'),
    _ = require('lodash'),
    fs = require('fs');

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