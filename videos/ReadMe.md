# Playback and Segmentation

## Segmentation
The goal of this task is to segment video frames in order to separate the foreground from the background. As i analyse the video, the background is fix and can slighly vary due to shadows. I therfore apply difference matting on the frames. The background is estimated as being the image mean over the first 30 frames as they not change. In order to produce much more accurate results, one would consider the matting problem or also condider complex transformations such as dilatation, erosion, morphological operations.

Run
```bash
python segmentation.py -m <method> -d <destination_file> -v <video_file>
```
Example
```bash
python segmentation.py -m 'SIMPLE' -v ../video_1.mp4
```
or simply
```bash
python segmentation.py -v ../video_1.mp4
```

## Playback
The goal of this task is to play a given video file name soundless. The approach introduced in this file consists into separating the preprocessing from the main rendering loop as much to approach real time features. As this code would not probably run on a RTOS, the number of frame per second might sightly defviate from the real one, but this is nearly not remaquable as watching the played video.
Notes: As i use polling for querying if a key has been pressed, some requests might not be run !!!. For later development switch to events.

Run
```bash
python playback.py -m -f <fps> -r <resolution> -v <video_file>
```
Example
```bash
python playback.py -m -f 15 -r '(300, 400)' -v ../video_1.mp4
```
