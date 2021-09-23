
/*
import * as tf from '../../node_modules/@tensorflow/tfjs';
import * as speechCommands from '../../@node_modulestensorflow-models/speech-commands';
*/
// https://teachablemachine.withgoogle.com/models/TZrro9spJ/
const URL = window.location.protocol + '//' + window.location.host + '/model/'
const MIN = 0.66
const nbs = [];
// let nbEuh = 0
// let nbYolo = 0

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
    const scoreContainer = document.getElementById("score-container");
    for (let i = 0; i < classLabels.length; i++) {
        labelContainer.appendChild(document.createElement("div"));
        scoreContainer.appendChild(document.createElement("div"));
        nbs[i] = 0
        labelContainer.childNodes[0].innerHTML = "initialized"
    }


    // listen() takes two arguments:
    // 1. A callback function that is invoked anytime a word is recognized.
    // 2. A configuration object with adjustable fields
    const last = []
    recognizer.listen(result => {

        const scores = result.scores; // probability of prediction for each class
        // render the probability scores per class
        for (let i = 0; i < classLabels.length; i++) {
            const label = classLabels[i]
            const score = scores[i].toFixed(2)
            /*
            const classPrediction = label + ": " + score;
            labelContainer.childNodes[i].innerHTML = classPrediction;
            */
            // on compte pas la classe de base
            if (i > 0 && score >= MIN) {
                // il faut qu'il y ait une rupture dans la voix (sinon un euhhh long compte plusieurs fois)
                if (last[i] != label) {
                    // if (label === "Euuh") {
                    //     ++nbEuh                        
                    // } else if (label == "Yolo") {
                    //     ++nbYolo
                    // }
                    nbs[i] = nbs[i] + 1
                    last[i] = label

                    const classScore = label + ": " + nbs[i];
                    scoreContainer.childNodes[i].innerHTML = classScore;

                    if(label === "Euuh") {                        
                        labelContainer.childNodes[0].innerHTML = "<h1>🐮</h1>"
                    } else if (label == "Yolo") {
                        labelContainer.childNodes[0].innerHTML = "<h1>🤪</h1>"
                    }
                }
            } else {
                last[i] = ""
            }
        }
    }, {
        includeSpectrogram: false, // in case listen should return result.spectrogram
        probabilityThreshold: 0.70, // (0.75)
        invokeCallbackOnNoiseAndUnknown: true,
        overlapFactor: 0.60 // (0.5) probably want between 0.5 and 0.75. More info in README
    });

    // Stop the recognition in 5 seconds.
    // setTimeout(() => recognizer.stopListening(), 5000);
}

tf.setBackend('wasm').then(() => init());

//init()