'use strict';

var webpack = require('webpack');

module.exports = {
    entry: './frontend/index',
    output: {
        filename: './public/index.js',
        library: 'mind' // mind - public variable (everything that entry exports)
    },

    watch: true,

    devtool: 'source-map',

    plugins: [
        new webpack.ProvidePlugin({
            Highcharts: 'highcharts',
            jQuery: './vendors/jquery/jquery-2.2.3.min'
        })
    ]
};