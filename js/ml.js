
/*
import * as tf from '../../node_modules/@tensorflow/tfjs';
import * as speechCommands from '../../@node_modulestensorflow-models/speech-commands';
*/
// https://teachablemachine.withgoogle.com/models/TZrro9spJ/
const URL = window.location.protocol + '//' + window.location.host + (window.location.pathname === '/' ? '' : window.location.pathname) + '/model/';
const nbs = [];

const SPECTRORAM_TIME_MS = 1000;
const MIN_TIME_BEETWEEN_MS = 600;
const OVERLAP_FACTOR = MIN_TIME_BEETWEEN_MS * 1.0 / SPECTRORAM_TIME_MS;
const PROBABILITY_THRESHOLD = 0.9;

const MAP_MIN_PROBABILITY_PER_LABELS = {
    "Euuh": PROBABILITY_THRESHOLD,
    "Yolo": PROBABILITY_THRESHOLD,
    "Next": PROBABILITY_THRESHOLD
};

async function createModel() {
    const checkpointURL = URL + "model.json"; // model topology
    const metadataURL = URL + "metadata.json"; // model metadata

    const recognizer = speechCommands.create(
        "BROWSER_FFT", // fourier transform type, not useful to change
        undefined, // speech commands vocabulary feature, not useful for your models
        checkpointURL,
        metadataURL);

    // check that model and metadata are loaded via HTTPS requests.
    await recognizer.ensureModelLoaded();

    return recognizer;
}

async function init() {
    const recognizer = await createModel();
    const classLabels = recognizer.wordLabels(); // get class labels
    const labelContainer = document.getElementById("label-container");
    for (let i = 0; i < classLabels.length; i++) {
        labelContainer.appendChild(document.createElement("div"));
        nbs[i] = 0;
    }
    document.getElementById("score-container-final").innerHTML = "<h1>🐮 " + nbs[classLabels.indexOf("Euuh")] + " 🤪 " + nbs[classLabels.indexOf("Yolo")] + "</h1>";

    // listen() takes two arguments:
    // 1. A callback function that is invoked anytime a word is recognized.
    // 2. A configuration object with adjustable fields
    let last = Date.now();
    let lastLabel = "";
    recognizer.listen(result => {

        const scores = result.scores; // probability of prediction for each class

        const maxScore = Math.max.apply(Math, scores);
        const index = scores.indexOf(maxScore);
        const label = classLabels[index];

        if ((lastLabel != label || (last + MIN_TIME_BEETWEEN_MS * 2) < Date.now()) &&
            MAP_MIN_PROBABILITY_PER_LABELS[label] <= maxScore
           ) {
            if(label === "Euuh") {
                labelContainer.className = 'show';
                labelContainer.childNodes[0].innerHTML = "<h1>🐮</h1>";
            } else if (label === "Yolo") {
                labelContainer.className = 'show';
                labelContainer.childNodes[0].innerHTML = "<h1>🤪</h1>";
            } else if (label === "Next") {
                Reveal.next();
            }
            nbs[index]++;
            if (labelContainer.className == 'show') {
                setTimeout(function() {
                    labelContainer.className = 'hide';
                }, MIN_TIME_BEETWEEN_MS);
            }
            document.getElementById("score-container-final").innerHTML = "<h1>🐮 " + nbs[classLabels.indexOf("Euuh")] + " 🤪 " + nbs[classLabels.indexOf("Yolo")] + "</h1>";
        }
        last = Date.now();
        lastLabel = label;
    }, {
        includeSpectrogram: false, // in case listen should return result.spectrogram
        probabilityThreshold: PROBABILITY_THRESHOLD, // (0.75)
        invokeCallbackOnNoiseAndUnknown: true,
        overlapFactor: OVERLAP_FACTOR // (0.5) probably want between 0.5 and 0.75. More info in README
    });

}

// Only on slide, not on speaker note
// (hack) on speaker node url contains parameter "?receiver..."
if (window.location.toString().indexOf("?receiver") == -1) {
    alreadyLoaded = 1
    tf.setBackend('wasm').then(() => init());
    //init() // default backend
}
