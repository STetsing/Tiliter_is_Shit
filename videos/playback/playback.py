''' The goal of this file is to play a given video file name soundless. The approach introduced in this file consists into
    separating the preprocessing from the main rendering loop as much to approach real time features. As this code would not probably run on a RTOS, the number of frame
    per second might sightly defviate from the real one, but this is nearly not remaquable as watching the played video.
    Notes: As i use polling for querying if a key has been pressed, some requests might not be run !!!. For later development switch to events.
    '''
# This code has be been run on MACos

import cv2
import numpy as np
import argparse
import sys
import os

def parse_arguments(argv):
    '''Parse sytem arguments'''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-v', '--video_file_path', type=str, help='the video path that will be played')
    parser.add_argument('-r', '--display_resolution', type=str, help='the display resolution. This should be in format (x, x)')
    parser.add_argument('-f', '--fps', type=int, help='Number of frame per second')
    parser.add_argument('-m', '--monochrome', action='store_true', help='States if the video should be monochrome')
    return parser.parse_args(argv)

def playVideo(filename, resolution, fps, monochrome):
    ''' Play the video provided by the file name'''
    try:
        #load de the file
        video = cv2.VideoCapture(filename)
        
        # No RTOS. But the number of frames will be close to the real time ones
        fps_frac = int(1/fps * 1000) # in ms
    
        if(video.isOpened()):
            # number total of frames
            n_frames  = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print('Reading the frames ...')
            frames = []
            for n in range(n_frames):
                ret, frame = video.read()
                
                #convert to gray scale
                if(monochrome):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
                #resize the frame
                frame = cv2.resize(frame, resolution, interpolation = cv2.INTER_NEAREST)
                
                frames.append(frame)
        
            print('Done reading the frames')
            print('Rendering the frames ...')

            #Render the frames
            paused = False
            f = 0
            while f < (len(frames)):
                # rendering not paused
                if not paused:
                    cv2.imshow('Frame', frames[f])
                    f += 1
                
                # pause the rendering
                if cv2.waitKey(fps_frac - 1) & 0xFF == ord('p'):
                    paused = True
                    while(paused):
                        # when paused, go back one frame
                        if cv2.waitKey(1) & 0xFF == ord('b'):
                            f -= 1
                            cv2.imshow('Frame', frames[f])
                        
                        # resume rendering
                        if cv2.waitKey(1) & 0xFF == ord('r'):
                            paused = False
                
                # quit rendering !!! waitkey 1 !!!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
        # release resources
        video.release()
        print('Done rendering the frames')


    except Exception as e:
        print('An error occured while reading the video file:', e)

def main(args):
    '''Main '''
    
    # Setup default config
    if args.video_file_path == None:
        print('Pease provide a video path !')
        return
    
    fps = 10 if args.fps == None else args.fps
    mono= False if args.monochrome == None else True
    res = (300, 300) if args.display_resolution == None else eval(args.display_resolution)
    
    # play back
    playVideo(args.video_file_path, res, fps, mono)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

