module.exports = function (config) {
  config.set({
    basePath: '',
    frameworks: ['jasmine'],
    files: ['bundle/tests/browser/*.js', 'bundle/tests/shared/*.js'],
    reporters: ['progress'],
    port: 9876,
    colors: true,
    logLevel: config.LOG_INFO,
    autoWatch: false,
    browsers: ['ChromeHeadless'],
    singleRun: true,
    concurrency: Infinity,
  });
};
