import * as tf from '@tensorflow/tfjs';
import * as tfd from '@tensorflow/tfjs-data';
import * as blazeface from '@tensorflow-models/blazeface';

let video, videoWidth, videoHeight, webcam;
async function setupCamera() {
  video = document.getElementById('video');
  // try {
  //   webcam = await tf.data.webcam(video);
  // } catch (e) {
  //   console.error("init webcam:", e);
  // }

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': { facingMode: 'user' },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

let canvas, canvasCtx;
async function setupCanvas() {
  canvas = document.getElementById('output');
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  canvasCtx = canvas.getContext('2d');
  canvasCtx.fillStyle = "rgba(255, 0, 0, 0.5)";
}

let faceDetectionModel;
async function loadFaceDetectionModel() {
  faceDetectionModel = await blazeface.load();
}

let maskDetectionModel;
async function loadMaskDetectionModel() {
  maskDetectionModel = await tf.loadLayersModel('./model.json');
}

async function getImage() {
  const img = await webcam.capture();
  // const processedImg = tf.tidy(() => img.expandDims(0).toFloat().div(127).sub(1));
  // img.dispose();
  return img;
}

async function renderPrediction() {
  const returnTensors = false;
  const flipHorizontal = true;
  const annotateBoxes = false;

  let faces;
  try {
    faces = await faceDetectionModel.estimateFaces(video, returnTensors, flipHorizontal, annotateBoxes);
  } catch (e) {
    console.error("estimateFaces:", e);
    return;
  }
  // console.log("faces", faces);

  if (faces.length > 0) {
    for (let i = 0; i < faces.length; i++) {
      let face = tf.browser.fromPixels(video)
        .resizeNearestNeighbor([224, 224])
        .toFloat();
      // face = tf.image.resizeBilinear(face, [224, 224]);
      // face = tf.cast(face, 'float32');
      // face = tf.tensor4d(Array.from(face.dataSync()), [1, 224, 224, 3])
      let offset = tf.scalar(127.5);
      face = face.sub(offset)
        .div(offset)
        .expandDims();

      let predictions;
      try {
        predictions = await maskDetectionModel.predict(face).data();
      } catch (e){
        console.log("maskDetection:", e);
        return;
      }

      face.dispose();

      const start = faces[i].topLeft;
      const end = faces[i].bottomRight;
      const size = [end[0] - start[0], end[1] - start[1]];

      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      canvasCtx.fillStyle = "rgba(255, 0, 0, 0.5)";
      if (predictions.length > 0 && predictions[0] > predictions[1]) {
        canvasCtx.fillStyle = "rgba(0, 255, 0, 0.5)";
      }
      canvasCtx.fillRect(start[0], start[1], size[0], size[1]);
    }
    // return;
  }

  requestAnimationFrame(renderPrediction);
}

async function main() {
  await setupCamera();
  video.play();

  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  setupCanvas();

  await loadFaceDetectionModel();
  await loadMaskDetectionModel();

  renderPrediction();
}

main();
