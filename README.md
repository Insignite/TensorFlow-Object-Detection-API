# Multiple Cards 9 to A Object Detection with TensorFlow (GPU) on Windows 10

## Summary

This repository is a personal project aiming to understand and learn TensorFlow's Object Detection API by
training an object detection classifier for multiple objects detection on Windows (This was test on Windows 10).
The repository contain both project code for card detection under "./workspace/training_demo/" directory and the entire TensorFlow Object Detection API. Beause card dataset 
was used in the training process, the frozen tensor graph will only detect card that similar in the dataset. However, 
you can import your own dataset and retrain the SSD_Inception_v2 model to recognize whatever your dataset contained.

Througout the tutorial, I have a lot of trouble getting environment variable path right in order for TensorFlow to run.
Here are some datapath that might fix the problem (not include the required CUDA path):
```
N:\TensorFlow\models\research\object_detection
N:\TensorFlow\models\research\object_detection\utils
N:\TensorFlow\models\research\slim
N:\TensorFlow\models\research\slim\datasets
N:\TensorFlow\models\research\slim\deployment
N:\TensorFlow\models\research\slim\nets
N:\TensorFlow\models\research\slim\preprocessing
N:\TensorFlow\models\research\slim\scripts
C:\Program Files\Google Protobuf\bin

```

__Dataset and project idea credit to:__ [_EdjeElectronics_](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)

__Step by step tutorial that I follow:__ [_TensorFlow Object Detection API tutorial_](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/)


## Result

This is final result archieving after 35,000 steps and take roughly 13 hours to train. SSD model have decent speed recognition but the trade off is accuracy. The model has easy time recognize 9, 10, A, and Jack, as the result below show. It clearly show that the model doesn't have low enough loss to define between Jack, Queen, and king even with 35,000 steps. _Real-time recognition_ is also available by running:
```
(Activate tensorflow_gpu if you has yet to do so and nagivate to __.\workspace\training_demo__ folder)
python webcamDetection.py
```

### Image Detection
![Result image of imageDetection.py](https://github.com/Insignite/TensorFlow-Object-Detection-API/blob/master/workspace/training_demo/resultDisplay/pictureDetectionResult.PNG)

### Video Detection
![Result video of videoDetection.gif](https://github.com/Insignite/TensorFlow-Object-Detection-API/blob/master/workspace/training_demo/resultDisplay/VideoDetectionGIF.gif)
