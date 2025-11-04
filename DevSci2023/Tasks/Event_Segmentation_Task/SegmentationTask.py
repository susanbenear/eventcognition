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


#create GUI box to input pp number and version
myDlg = gui.Dlg()
myDlg.addField('Participant ID:')
myDlg.addField('Version:')
myDlg.show()
#convert these values to strings and save them to variables
participant = str(myDlg.data[0])
version = str(myDlg.data[1])
#get date timestamp
date = data.getDateStr()

#set paths for various directories to be used later
project_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_dir)
video_dir = 'Version_'+version+'/videos'
data_dir='data/'

#define lists, variables, etc.
stimulus_key = pd.read_csv('Version_'+version+'/videosheet.csv')
video_clip = stimulus_key['videoclip']
breakimage = 'Version_'+version+'/kjstill.png'
r = open('text_instructions.txt', encoding='utf8')
instr = r.read().splitlines()
r.close()
allLetters = list(string.ascii_lowercase)
time_break = 5
kb = keyboard.Keyboard()

#Set up variables for the Experiment Handler
expName = 'learningandlemurs'
expInfo = {'participant': participant, 'version': version}
expInfo['date'] = date
expInfo['expName'] = expName
filename = project_dir + os.sep + u'data/%s_%s_%s_%s' % (expInfo['participant'], expName, expInfo['version'], expInfo['date'])

# Set up the ExperimentHandler
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=None, runtimeInfo=None,
    originPath=project_dir,
    savePickle=True, saveWideText=True,
    dataFileName=filename, autoLog=True)

#add variables to the experiment file
thisExp.addData('participant', participant)
thisExp.addData('version', version)
thisExp.addData('date', date)

# START TASK

# create window 
win = visual.Window(allowGUI=True, monitor='testMonitor', 
                    units='norm', color="Gray", fullscr=True)

# close experiment when esc is pressed
if event.getKeys(['escape']):
    win.close()
    core.quit()

# INSTRUCTIONS
#present instructions by indexing a particular line from a text file
temp_instr = visual.TextStim(win, instr[0], color='black', pos=(0.0, 0.0))
#pull this text onto the screen
temp_instr.draw()
win.update()
#advance the screen to the next only when the enter/return key is pressed
event.waitKeys(keyList=['return'])
win.flip()

# PRESENT VIDEOS
#set n to 0 to make the while loop pull the "0th" (1st) video first
n = 0
while n != 6: #loop will play videos 0-5 a.k.a. all 6 video clips
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
    # create an empty list to later append each RT for each video to
    RTlist = []
    
    # while the video plays...
    while not mov.isFinished:
        keys = kb.getKeys(['space', 'escape']) #define allowed keys
        for thisKey in keys:
            if thisKey=='escape': #if escape selected, quit the program
                mov.stop()
                core.quit()
            # if space selected, add the RT to RTlist for this video
            elif thisKey=='space': 
                RT = kb.clock.getTime() #RT defined in s by the keyboard clock
                RTlist.append(RT)
        # add the list of RTs to the .csv in its own column titled by the video
        # if space selected, add response time to the csv
        thisExp.addData('RTvid'+str(video_clip[n]), RTlist) 
        # draw elements
        mov.draw() #bring the movie onto the screen
        win.flip() 
    
    # BREAK SCREEN
    break_instr = visual.TextStim(win, instr[1], color='black', pos=[0,0.45])
    break_image = visual.ImageStim(win, breakimage, pos=[0,-0.3])
    break_instr.draw()
    break_image.draw()
    win.update()
    event.waitKeys(keyList=['return'])
    win.flip()

    n = n + 1

#EXIT SCREEN
exit_instr = visual.TextStim(win, instr[2], color='black', pos=[0,0])
exit_instr.draw()
win.update()
event.waitKeys(keyList=['return'])
win.flip()

win.close()
core.quit()
