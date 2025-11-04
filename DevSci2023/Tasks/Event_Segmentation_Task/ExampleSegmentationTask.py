#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from psychopy import visual, core, event, data, gui, logging
from psychopy.hardware import keyboard
import os
import pandas as pd
from PIL import Image
import numpy as np
import string
import sys

#set paths for various directories to be used later
project_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_dir)
video_dir = 'ExampleVids'

# create window 
win = visual.Window(allowGUI=True, monitor='testMonitor', 
                    units='norm', color="Gray", fullscr=True)

#define lists, variables, etc.
stimulus_key = pd.read_csv('ExampleVids'+'/examplevideosheet.csv')
video_clip = stimulus_key['videoclip']
kb = keyboard.Keyboard()

# START TASK

# close experiment when esc is pressed
if event.getKeys(['escape']):
    win.close()
    core.quit()
    
# INTRO TEXT
keystextintro = "Welcome! Let's pratice pressing the button for each event!"
text = visual.TextStim(win, keystextintro, color='black', height=50, pos=(0, 0), units='pix')
text.draw()
win.update()
event.waitKeys(keyList=['return'])
win.flip()
    
# PRESENT VIDEOS
#set n to 0 to make the while loop pull the "0th" (1st) video first
n = 0
while n != 2: #loop will play videos 0-1 a.k.a. both video clips
    video_selected = project_dir+'/'+video_dir+'/'+video_clip[n] #selects each video clip
    #set path for video
    videopath = str(video_selected) 
    if not os.path.exists(videopath):
        raise RuntimeError("Video File could not be found:" + videopath)

    #play the video using VLCMovieStim
    mov = visual.VlcMovieStim(win, videopath,
        size=[960,540],  # set as `None` to use the native video size
        pos=[0, 0],  
        flipVert=False, 
        flipHoriz=False, 
        loop=False,  
        autoStart=True)  
    
    # reset the keyboard clock    
    kb.clock.reset()
    #create an empty list to later append each RT for each video to
    RTlist = []
    
    #while the video plays...
    while not mov.isFinished:
        keys = kb.getKeys(['space', 'escape']) #define allowed keys
        for thisKey in keys:
            if thisKey=='escape': #if escape selected, quit the program
                mov.stop()
                core.quit()
            elif thisKey=='space': #if space selected, add the RT to RTlist for this video
                RT = kb.clock.getTime() #RT defined in s by the keyboard clock
                RTlist.append(RT)#while the video plays...
        # draw elements
        mov.draw() #bring the movie onto the screen
        win.flip() 
    
    # BREAK SCREEN
    keystext = "That was fun! Now it's your turn to try!"
    text = visual.TextStim(win, keystext, color='black', height=50, pos=(0, 0), units='pix')
    text.draw()
    win.update()
    event.waitKeys(keyList=['return'])
    win.flip()

    n = n + 1

win.close()
core.quit()
