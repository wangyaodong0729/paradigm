# -*- coding: utf-8 -*-

import os
import numpy as np
from math import pi
from psychopy import data, visual, event, sound
from metabci.brainstim.utils import NeuroScanPort, _check_array_like
import threading
from neuracle_lib.triggerBox import TriggerBox,TriggerIn,PackageSensorPara
import time
import pickle

isTriggerIn = True
isTriggerBox = False
# preFunctions
def sinusoidal_sample(freqs, phases, srate, frames, stim_color):
    """Sinusoidal approximate sampling method.
    -author: Qiaoyi Wu
    -Created on: 2022-06-20
    -update log:
        2022-06-26 by Jianhang Wu
        2022-08-10 by Wei Zhao
    Parameters
    ----------
        freqs: list of float,
            Frequencies of each stimulus.
        phases: list of float,
            Phases of each stimulus.
        srate: int or float,
            Refresh rate of screen.
        frames: int,
            Flashing frames.
        stim_color: list,
            Color of stimu.
    Returns
    ----------
        color: ndarray,
            (n_frames, n_elements, 3)
    """

    time = np.linspace(0, (frames - 1) / srate, frames)
    color = np.zeros((frames, len(freqs), 3))
    for ne, (freq, phase) in enumerate(zip(freqs, phases)):
        sinw = np.sin(2 * pi * freq * time + pi * phase) + 1
        color[:, ne, :] = np.vstack(
            (sinw * stim_color[0], sinw * stim_color[1], sinw * stim_color[2])
        ).T
        if stim_color == [-1, -1, -1]:
            pass
        else:
            if stim_color[0] == -1:
                color[:, ne, 0] = -1
            if stim_color[1] == -1:
                color[:, ne, 1] = -1
            if stim_color[2] == -1:
                color[:, ne, 2] = -1

    return color


# 这是我自己定义的范式，SSVEP和MI混合范式
class MyStim1(object):

    def __init__(self, win, colorSpace='rgb', allowGUI=True):
        self.freqs = None
        self.phases = None
        self.stim_contrs = None
        self.stim_sfs = None
        self.stim_oris = None
        self.refresh_rate = None
        self.stim_frames = None
        self.stim_opacities = None
        self.stim_time = None
        self.index_color = None
        self.index_color1 = None
        self.stim_color = None
        self.stim_sizes = None
        self.stim_pos = None
        self.right_pos = None
        self.left_pos = None
        self.up_pos = None
        self.down_pos = None
        self.n_Elements = None
        self.stim_width = None
        self.stim_length = None
        self.stim_colors = None
        self.flash_stimuli = None
        self.response_back = None
        self.response = None
        self.left_index = None
        self.right_index = None
        self.gift_index =None
        self.miner_index = None
        self.right_stimuli = None
        self.left_stimuli = None
        self.win = win
        win.colorSpace = colorSpace
        win.allowGUI = allowGUI
        win_size = win.size
        self.win_size = np.array(win_size)
        self.n_elements = 2

        self.tex_left = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'textures\\golden.png')
        self.tex_right = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'textures\\golden.png')
        self.tex_gift = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'textures\\gift.png')
        # self.tex_girl = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                              'textures\\girls.png')
        self.tex_miner = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'textures\\golden_miner.png')
        self.tex_cue = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'textures\\cue.wav')
        self.tex_fail = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'textures\\fail.wav')
        self.tex_success = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'textures\\success.wav')
        self.tex_happy = os.path.join(os.path.abspath(os.path.dirname(os.path.abspath(__file__))),
                                     'textures\\happy.png')
        self.tex_sad = os.path.join(os.path.abspath(os.path.dirname(os.path.abspath(__file__))),
                                      'textures\\sad.png')

    # 配置左右手的提示q
    def config_stim(self, left_pos=None, right_pos=None, gift_pos=None,miner_pos=None,stim_length=288, stim_width=288,
                    n_Elements=1, stim_color=None, index_color=None,index_color1=None):
        # 设置刺激的大小
        if index_color is None:
            index_color = [[-1, -1, -1]]
        if index_color1 is None:
            index_color1 = [[-1, -1, -1]]
        if stim_color is None:
            stim_color = [[-1, -1, -1]]
        if right_pos is None:
            right_pos = [480, 0.0]
        if left_pos is None:
            left_pos = [-480, 0.0]
        if gift_pos is None:
            gift_pos = [0, 0.0]
        if miner_pos is None:
            miner_pos = [0, 200.0]

        miner_length = 387
        miner_width = 432
        self.stim_length = stim_length
        self.stim_width = stim_width
        self.n_Elements = n_Elements
        # 设置刺激的位置
        self.left_pos = left_pos
        self.right_pos = right_pos
        self.stim_pos = np.array([left_pos, right_pos])
        # check sizeof stimuli
        stim_sizes  = np.zeros((2, 2))
        stim_sizes[:] = np.array([stim_length, stim_width])
        self.stim_sizes = stim_sizes
        # 设置刺激颜色
        self.stim_color = stim_color
        self.index_color = index_color
        self.index_color1 = index_color1

        self.left_index = visual.ElementArrayStim(self.win, units='pix', elementTex=self.tex_left,
                                                  elementMask=None, texRes=2, nElements=n_Elements,
                                                  sizes=[[stim_length, stim_width]], xys=np.array([left_pos]),
                                                  oris=[0], colors=np.array(self.index_color), opacities=[1],
                                                  contrs=[-1])

        self.right_index = visual.ElementArrayStim(self.win, units='pix', elementTex=self.tex_right,
                                                   elementMask=None, texRes=2, nElements=n_Elements,
                                                   sizes=[[stim_length, stim_width]], xys=np.array([right_pos]),
                                                   oris=[0], colors=np.array(self.index_color1), opacities=[1],
                                                   contrs=[-1])

        self.left_stimuli = visual.ElementArrayStim(self.win, units='pix', elementTex=self.tex_left,
                                                    elementMask=None, texRes=2, nElements=n_Elements,
                                                    sizes=[[stim_length, stim_width]], xys=np.array([left_pos]),
                                                    oris=[0], colors=np.array(self.stim_color), opacities=[0.1],
                                                    contrs=[-1])
        self.right_stimuli = visual.ElementArrayStim(self.win, units='pix', elementTex=self.tex_right,
                                                     elementMask=None, texRes=2, nElements=n_Elements,
                                                     sizes=[[stim_length, stim_width]], xys=np.array([right_pos]),
                                                     oris=[0], colors=np.array(self.stim_color), opacities=[0.1],
                                                     contrs=[-1])
        self.gift_index = visual.ElementArrayStim(self.win, units='pix', elementTex=self.tex_gift,
                                                   elementMask=None, texRes=2, nElements=n_Elements,
                                                   sizes=[[stim_length, stim_width]], xys=np.array([gift_pos]),
                                                   oris=[0], colors=np.array(self.stim_color), opacities=[1],
                                                   contrs=[-1])
        self.happy_index = visual.ElementArrayStim(self.win, units='pix', elementTex=self.tex_happy,
                                                   elementMask=None, texRes=2, nElements=n_Elements,
                                                   sizes=[[stim_length, stim_width]], xys=np.array([gift_pos]),
                                                   oris=[0], colors=np.array(self.stim_color), opacities=[1],
                                                   contrs=[-1])
        self.sad_index = visual.ElementArrayStim(self.win, units='pix', elementTex=self.tex_sad,
                                                   elementMask=None, texRes=2, nElements=n_Elements,
                                                   sizes=[[stim_length, stim_width]], xys=np.array([gift_pos]),
                                                   oris=[0], colors=np.array(self.stim_color), opacities=[1],
                                                   contrs=[-1])
        self.miner_index = visual.ElementArrayStim(self.win, units='pix', elementTex=self.tex_miner,
                                          elementMask=None, texRes=2, nElements=n_Elements,
                                          sizes=[[miner_length, miner_width]], xys=np.array([miner_pos]),
                                          oris=[0], colors=np.array(self.stim_color), opacities=[1],
                                          contrs=[-1])
        self.cue = sound.Sound(self.tex_cue) 
        self.success = sound.Sound(self.tex_success)
        self.fail = sound.Sound(self.tex_fail)
        # self.mov2 = visual.MovieStim(self.win, elementTex=self)
    # 配置SSVEP闪烁刺激
    def config_flash(self, refresh_rate, stim_time, flash_color, stimtype='sinusoid', stim_opacities=1, **kwargs):
        self.refresh_rate = refresh_rate
        self.stim_time = stim_time
        self.stim_opacities = stim_opacities
        self.stim_frames = int(stim_time * self.refresh_rate)

        if refresh_rate == 0:
            self.refresh_rate = np.floor(self.win.getActualFrameRate(nIdentical=20, nWarmUpFrames=20))

        self.stim_oris = np.zeros((self.n_elements,))  # orientation
        self.stim_sfs = np.zeros((self.n_elements,))  # spatial frequency
        self.stim_contrs = np.ones((self.n_elements,))  # contrast

        # check extra inputs
        if 'stim_oris' in kwargs.keys():
            self.stim_oris = kwargs['stim_oris']
        if 'stim_sfs' in kwargs.keys():
            self.stim_sfs = kwargs['stim_sfs']
        if 'stim_contrs' in kwargs.keys():
            self.stim_contrs = kwargs['stim_contrs']
        if 'freqs' in kwargs.keys():
            self.freqs = kwargs['freqs']
        if 'phases' in kwargs.keys():
            self.phases = kwargs['phases']

        # 生成正弦控制信号
        if stimtype == 'sinusoid':
            self.stim_colors = sinusoidal_sample(freqs=self.freqs, phases=self.phases, srate=self.refresh_rate,
                                                 frames=self.stim_frames, stim_color=flash_color)
            if self.stim_colors[0].shape[0] != self.n_elements:
                raise Exception('Please input correct num of stims!')

        # 检查stim_colors的尺寸是否合法
        incorrect_frame = (self.stim_colors.shape[0] != self.stim_frames)
        incorrect_number = (self.stim_colors.shape[1] != self.n_elements)
        if incorrect_frame or incorrect_number:
            raise Exception('Incorrect color matrix or flash frames!')

        self.flash_stimuli = []
        for sf in range(self.stim_frames):
            self.flash_stimuli.append(visual.ElementArrayStim(win=self.win, units='pix', nElements=self.n_elements,
                                                              sizes=self.stim_sizes, xys=self.stim_pos,
                                                              colors=self.stim_colors[sf, ...],
                                                              opacities=self.stim_opacities,
                                                              oris=self.stim_oris, sfs=self.stim_sfs,
                                                              contrs=self.stim_contrs,
                                                              elementTex=np.ones((64, 64)), elementMask=None,
                                                              texRes=48))

    def config_response(self, response_pos=None, response_color_back=None, response_color=None):
        # 配置大小
        if response_color_back is None:
            response_color_back = [[0.5, 0.5, 0.5]]
        if response_color is None:
            response_color = [[1, 1, 1]]
        if response_pos is None:
            response_pos = [0.0, 0.0]
        response_pos = np.array([response_pos])

        self.response_back = visual.Rect(win=self.win, width=100, height=500, units='pix',
                                              fillColor=response_color_back,
                                              pos=response_pos)
        response_pos[0][1] = -10
        self.response = visual.Rect(win=self.win, width=100, height=0, units='pix', fillColor=response_color,
                                         pos=response_pos)

class MyStim2(object):

    def __init__(self, win, colorSpace='rgb', allowGUI=True):
        self.freqs = None
        self.phases = None
        self.stim_contrs = None
        self.stim_sfs = None
        self.stim_oris = None
        self.refresh_rate = None
        self.stim_frames = None
        self.stim_opacities = None
        self.stim_time = None
        self.index_color = None
        self.index_color1 = None
        self.stim_color = None
        self.stim_sizes = None
        self.stim_pos = None
        self.hammer_pos = None
        self.right_pos = None
        self.left_pos = None
        self.up_pos = None
        self.down_pos = None
        self.n_Elements = None
        self.stim_width = None
        self.stim_length = None
        self.stim_colors = None
        self.flash_stimuli = None
        self.response_back = None
        self.response = None
        self.left_index = None
        self.right_index = None
        self.up_index = None
        self.down_index = None
        self.gift_index =None
        self.right_stimuli = None
        self.left_stimuli = None
        self.win = win
        win.colorSpace = colorSpace
        win.allowGUI = allowGUI
        win_size = win.size
        self.win_size = np.array(win_size)
        self.n_elements = 2

        self.tex_mouse = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'textures\\mouse.png')
        self.tex_basin = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'textures\\basin.png')
        self.tex_hammer = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'textures\\hammer.png')
        self.tex_gift = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'textures\\gift.png')
        self.tex_cue = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'textures\\cue.wav')
        self.tex_fail = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'textures\\fail.wav')
        self.tex_success = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'textures\\success.wav')
        self.tex_happy = os.path.join(os.path.abspath(os.path.dirname(os.path.abspath(__file__))),
                                     'textures\\happy.png')
        self.tex_sad = os.path.join(os.path.abspath(os.path.dirname(os.path.abspath(__file__))),
                                      'textures\\sad.png')

    # 配置左右手的提示q
    def config_stim(self, left_pos=None, right_pos=None, up_pos=None, down_pos=None, left_p=None, right_p=None, up_p=None, down_p=None, gift_pos=None,hammer_pos=None, mouse_length=300, mouse_width=280,stim_length=300, stim_width=230,
                    n_Elements=1, stim_color=None, index_color=None,index_color1=None):
        # 设置刺激的大小
        if index_color is None:
            index_color = [[-1, -1, -1]]
        if index_color1 is None:
            index_color1 = [[-1, -1, -1]]
        if stim_color is None:
            stim_color = [[-1, -1, -1]]
        if right_pos is None:
            right_pos = [400, 0.0]
        if left_pos is None:
            left_pos = [-400, 0.0]
        if up_pos is None:
            up_pos = [0.0, 300]
        if down_pos is None:
            down_pos = [0.0, -300]
        if gift_pos is None:
            gift_pos = [0, 0.0]
        if hammer_pos is None:
            hammer_pos = [0, 0]
        cha = (mouse_width - stim_width)/2    
        if right_p is None:
            right_p = [400, 0 + cha]
        if left_p is None:
            left_p = [-400, 0 + cha]
        if up_p is None:
            up_p = [0.0, 300 + cha]
        if down_p is None:
            down_p = [0.0, -300 + cha]

        hammer_length = 300
        hammer_width = 260
        self.stim_length = stim_length
        self.stim_width = stim_width
        self.n_Elements = n_Elements
        # 设置刺激的位置
        self.left_pos = left_pos
        self.right_pos = right_pos
        self.up_pos = up_pos
        self.down_pos = down_pos
        self.stim_pos = np.array([left_pos, right_pos])
        # check sizeof stimuli
        stim_sizes  = np.zeros((2, 2))
        stim_sizes[:] = np.array([stim_length, stim_width])
        self.stim_sizes = stim_sizes
        # 设置刺激颜色
        self.stim_color = stim_color
        self.index_color = index_color
        self.index_color1 = index_color1

        self.left_index = visual.ElementArrayStim(self.win, units='pix', elementTex=self.tex_mouse,
                                                  elementMask=None, texRes=2, nElements=n_Elements,
                                                  sizes=[[mouse_length, mouse_width]], xys=np.array([left_p]),
                                                  oris=[0], colors=np.array(self.index_color), opacities=[1],
                                                  contrs=[-1])

        self.right_index = visual.ElementArrayStim(self.win, units='pix', elementTex=self.tex_mouse,
                                                   elementMask=None, texRes=2, nElements=n_Elements,
                                                   sizes=[[mouse_length, mouse_width]], xys=np.array([right_p]),
                                                   oris=[0], colors=np.array(self.index_color1), opacities=[1],
                                                   contrs=[-1])
        self.up_index = visual.ElementArrayStim(self.win, units='pix', elementTex=self.tex_mouse,
                                                  elementMask=None, texRes=2, nElements=n_Elements,
                                                  sizes=[[mouse_length, mouse_width]], xys=np.array([up_p]),
                                                  oris=[0], colors=np.array(self.index_color), opacities=[1],
                                                  contrs=[-1])

        self.down_index = visual.ElementArrayStim(self.win, units='pix', elementTex=self.tex_mouse,
                                                   elementMask=None, texRes=2, nElements=n_Elements,
                                                   sizes=[[mouse_length, mouse_width]], xys=np.array([down_p]),
                                                   oris=[0], colors=np.array(self.index_color1), opacities=[1],
                                                   contrs=[-1])

        self.left_stimuli = visual.ElementArrayStim(self.win, units='pix', elementTex=self.tex_basin,
                                                    elementMask=None, texRes=2, nElements=n_Elements,
                                                    sizes=[[stim_length, stim_width]], xys=np.array([left_pos]),
                                                    oris=[0], colors=np.array(self.stim_color), opacities=[1],
                                                    contrs=[-1])
        self.right_stimuli = visual.ElementArrayStim(self.win, units='pix', elementTex=self.tex_basin,
                                                     elementMask=None, texRes=2, nElements=n_Elements,
                                                     sizes=[[stim_length, stim_width]], xys=np.array([right_pos]),
                                                     oris=[0], colors=np.array(self.stim_color), opacities=[1],
                                                     contrs=[-1])
        self.up_stimuli = visual.ElementArrayStim(self.win, units='pix', elementTex=self.tex_basin,
                                                    elementMask=None, texRes=2, nElements=n_Elements,
                                                    sizes=[[stim_length, stim_width]], xys=np.array([up_pos]),
                                                    oris=[0], colors=np.array(self.stim_color), opacities=[1],
                                                    contrs=[-1])
        self.down_stimuli = visual.ElementArrayStim(self.win, units='pix', elementTex=self.tex_basin,
                                                     elementMask=None, texRes=2, nElements=n_Elements,
                                                     sizes=[[stim_length, stim_width]], xys=np.array([down_pos]),
                                                     oris=[0], colors=np.array(self.stim_color), opacities=[1],
                                                     contrs=[-1])
        self.gift_index = visual.ElementArrayStim(self.win, units='pix', elementTex=self.tex_gift,
                                                   elementMask=None, texRes=2, nElements=n_Elements,
                                                   sizes=[[288, 288]], xys=np.array([gift_pos]),
                                                   oris=[0], colors=np.array(self.stim_color), opacities=[1],
                                                   contrs=[-1])
        self.happy_index = visual.ElementArrayStim(self.win, units='pix', elementTex=self.tex_happy,
                                                   elementMask=None, texRes=2, nElements=n_Elements,
                                                   sizes=[[288, 288]], xys=np.array([gift_pos]),
                                                   oris=[0], colors=np.array(self.stim_color), opacities=[1],
                                                   contrs=[-1])
        self.sad_index = visual.ElementArrayStim(self.win, units='pix', elementTex=self.tex_sad,
                                                   elementMask=None, texRes=2, nElements=n_Elements,
                                                   sizes=[[288, 288]], xys=np.array([gift_pos]),
                                                   oris=[0], colors=np.array(self.stim_color), opacities=[1],
                                                   contrs=[-1])
        self.hammer_index = visual.ElementArrayStim(self.win, units='pix', elementTex=self.tex_hammer,
                                          elementMask=None, texRes=2, nElements=n_Elements,
                                          sizes=[[hammer_length, hammer_width]], xys=np.array([hammer_pos]),
                                          oris=[0], colors=np.array(self.stim_color), opacities=[1],
                                          contrs=[-1])
        self.cue = sound.Sound(self.tex_cue) 
        self.success = sound.Sound(self.tex_success)
        self.fail = sound.Sound(self.tex_fail)
    # 配置SSVEP闪烁刺激
    def config_flash(self, refresh_rate, stim_time, flash_color, stimtype='sinusoid', stim_opacities=1, **kwargs):
        self.refresh_rate = refresh_rate
        self.stim_time = stim_time
        self.stim_opacities = stim_opacities
        self.stim_frames = int(stim_time * self.refresh_rate)

        if refresh_rate == 0:
            self.refresh_rate = np.floor(self.win.getActualFrameRate(nIdentical=20, nWarmUpFrames=20))

        self.stim_oris = np.zeros((self.n_elements,))  # orientation
        self.stim_sfs = np.zeros((self.n_elements,))  # spatial frequency
        self.stim_contrs = np.ones((self.n_elements,))  # contrast

        # check extra inputs
        if 'stim_oris' in kwargs.keys():
            self.stim_oris = kwargs['stim_oris']
        if 'stim_sfs' in kwargs.keys():
            self.stim_sfs = kwargs['stim_sfs']
        if 'stim_contrs' in kwargs.keys():
            self.stim_contrs = kwargs['stim_contrs']
        if 'freqs' in kwargs.keys():
            self.freqs = kwargs['freqs']
        if 'phases' in kwargs.keys():
            self.phases = kwargs['phases']

        # 生成正弦控制信号
        if stimtype == 'sinusoid':
            self.stim_colors = sinusoidal_sample(freqs=self.freqs, phases=self.phases, srate=self.refresh_rate,
                                                 frames=self.stim_frames, stim_color=flash_color)
            if self.stim_colors[0].shape[0] != self.n_elements:
                raise Exception('Please input correct num of stims!')

        # 检查stim_colors的尺寸是否合法
        incorrect_frame = (self.stim_colors.shape[0] != self.stim_frames)
        incorrect_number = (self.stim_colors.shape[1] != self.n_elements)
        if incorrect_frame or incorrect_number:
            raise Exception('Incorrect color matrix or flash frames!')

        self.flash_stimuli = []
        for sf in range(self.stim_frames):
            self.flash_stimuli.append(visual.ElementArrayStim(win=self.win, units='pix', nElements=self.n_elements,
                                                              sizes=self.stim_sizes, xys=self.stim_pos,
                                                              colors=self.stim_colors[sf, ...],
                                                              opacities=self.stim_opacities,
                                                              oris=self.stim_oris, sfs=self.stim_sfs,
                                                              contrs=self.stim_contrs,
                                                              elementTex=np.ones((64, 64)), elementMask=None,
                                                              texRes=48))

    def config_response(self, response_pos=None, response_color_back=None, response_color=None):
        # 配置大小
        if response_color_back is None:
            response_color_back = [[0.5, 0.5, 0.5]]
        if response_color is None:
            response_color = [[1, 1, 1]]
        if response_pos is None:
            response_pos = [0.0, 0.0]
        response_pos = np.array([response_pos])

        self.response_back = visual.Rect(win=self.win, width=100, height=500, units='pix',
                                              fillColor=response_color_back,
                                              pos=response_pos)
        response_pos[0][1] = -10
        self.response = visual.Rect(win=self.win, width=100, height=0, units='pix', fillColor=response_color,
                                         pos=response_pos)

class GetPlabel_MyTherad:
    """Start a thread that receives online results
    -author: Wei Zhao
    -Created on: 2022-07-30
    -update log:
        2022-08-10 by Wei Zhao
    Parameters
    ----------
    inlet:
        Stream data online.
    """

    def __init__(self):
        self._t_loop = None
        self._exit = threading.Event()

    def feedbackThread(self):
        """Start the thread."""
        self._t_loop = threading.Thread(
            target=self._inner_loop, name="get_predict_id_loop"
        )
        self._t_loop.start()

    def _inner_loop(self):
        """The inner loop in the thread."""
        self._exit.clear()

        while not self._exit.is_set():
            try:
                pass

            except Exception:
                pass

    def stop_feedbackThread(self):
        """Stop the thread."""
        self._exit.set()
        self._t_loop.join()


# basic experiment control
def paradigm(
        VSObject,
        win,
        bg_color,
        display_time=1.0,
        index_time=1.0,
        rest_time=0.5,
        port_addr="COM3",
        nrep=1,
        pdim="ssvep",
        lsl_source_id=None,
        online=None,
        # device_type='Neuracle'
):
    """Passing outsied parameters to inner attributes.
    -author: Wei Zhao
    -Created on: 2022-07-30
    -update log:
        2022-08-10 by Wei Zhao
        2022-08-03 by Shengfu Wen
        2022-12-05 by Jie Mei
    Parameters
    ----------
        VSObject: object of paradigm
        win: window
        bg_color: ndarray,
            Background color.
        display_time: float,
            Keyboard display time before 1st index.
        index_time: float,
            Indicator display time.
        rest_time: float, optional,
            Rest-state time.
        port_addr:
             Computer port.
        nrep: int,
            Num of blocks.
        pdim: name of paradigm
        lsl_source_id: str,
            Source id.
        online: bool,
            Flag of online experiment.
        device_type: str,
            See support device list in brainstim README file
    """
    if not _check_array_like(bg_color, 3):
        raise ValueError("bg_color should be 3 elements array-like object.")
    win.color = bg_color
    fps = VSObject.refresh_rate

    #用于发送标签
    # if device_type == 'NeuroScan':
    #      triggerin = NeuroScanPort(port_addr, use_serial=False) if port_addr else None
    # elif device_type == 'Neuracle':
    #
    #     port = NeuraclePort(port_addr) if port_addr else None
    #     triggerin = TriggerIn(port_addr) if port_addr else None
    #     flag = triggerin.validate_device()
    # else:
    #     raise KeyError("Unknown device type: {}, please check your input".format(device_type))

    port = TriggerIn(port_addr)
    port.validate_device()

    port_frame = int(0.05 * fps)

    if pdim == 'hybrid_bci1':
        # 设置实验设定，nrqep是blocks次数
        # 设置实验条件
        conditions = [
            {'index': 0, 'name': 'left_hand'},
            {'index': 1, 'name': 'right_hand'},
        ]

        trials = data.TrialHandler(conditions, nrep, name='experiment', method='random')

        # if flag:
        #     triggerin.output_event_data(0)

        count_trial = 0

        for trial in trials:

            count_trial += 1
            # quit demo
            # VSObject.girl_index.draw()
            VSObject.miner_index.draw()

            keys = event.getKeys(['q'])
            if 'q' in keys:
                break

            # episode 1: display interface
            iframe = 0  # 计数器
            while iframe < int(fps * display_time):

                # VSObject.girl_index.draw()
                VSObject.miner_index.draw()
                VSObject.left_stimuli.draw()
                VSObject.right_stimuli.draw()
                iframe += 1
                win.flip()
            # initialise index position
            index = int(trial['index'])
            if index == 0:
                left_tex = VSObject.left_index
                right_tex = VSObject.right_stimuli
            else:
                left_tex = VSObject.left_stimuli
                right_tex = VSObject.right_index

            # phase II: index(eye shifting)
            iframe = 0
            ref = VSObject.sad_index
            VSObject.cue.play()
            flag = False
            event.clearEvents()
            while iframe < int(fps * index_time):
                # if iframe == 0:

                    # VSObject.win.callOnFlip(port.output_event_data,index + 1)
                # VSObject.girl_index.draw()
                VSObject.miner_index.draw()
                left_tex.draw()
                right_tex.draw()

                keys = event.getKeys(['left', 'right'])
                if not flag and bool(keys):
                    flag =True
                    if index == 0 and keys[0] == 'left':
                        ref = VSObject.happy_index

                    elif index != 0 and keys[0] == 'right':
                        ref = VSObject.happy_index
                        
                iframe += 1
                win.flip()

            # phase III: target stimulating
            # port.output_event_data(0)
            # for sf in range(VSObject.stim_frames):
            #     # A

            #         # VSObject.win.callOnFlip(port.output_event_data,index + 1)

            #     # VSObject.girl_index.draw()
            #     VSObject.miner_index.draw()
            #     VSObject.left_stimuli.draw()
            #     VSObject.right_stimuli.draw()
            #     win.flip()

            # phase IV: respond
            iframe = 0
            respond_time = 1
            if ref == VSObject.happy_index:
                VSObject.success.play()
            elif ref == VSObject.sad_index:
                VSObject.fail.play()
            while iframe < int(fps * respond_time):
                ref.draw()
                iframe += 1
                win.flip()

            # phase I: rest state
            if count_trial % 5 == 0:
                rest_time = 5
            if count_trial % 10 == 0:
                rest_time = 10
            if rest_time != 0:
                iframe = 0
                while iframe < int(fps * rest_time):
                    VSObject.gift_index.draw()
                    # VSObject.left_stimuli.draw()
                    # VSObject.right_stimuli.draw()
                    iframe += 1
                    win.flip()
            rest_time = 1
    elif pdim == 'hybrid_bci2':
        # 设置实验设定，nrqep是blocks次数
        # 设置实验条件
        conditions = [
            {'index': 0, 'name': 'left'},
            {'index': 1, 'name': 'right'},
            {'index': 2, 'name': 'up'},
            {'index': 3, 'name': 'down'},
        ]
        trials = data.TrialHandler(conditions, nrep, name='experiment', method='random')
        # if flag:
        #     triggerin.output_event_data(0)
        count_trial = 0
        for trial in trials:
            count_trial += 1
            # quit demo
            VSObject.hammer_index.draw()
            VSObject.left_stimuli.draw()
            VSObject.right_stimuli.draw()
            VSObject.up_stimuli.draw()
            VSObject.down_stimuli.draw()
            keys = event.getKeys(['q'])
            if 'q' in keys:
                break
            # episode 1: display interface
            iframe = 0  # 计数器
            while iframe < int(fps * display_time):
                VSObject.hammer_index.draw()
                VSObject.left_stimuli.draw()
                VSObject.right_stimuli.draw()
                VSObject.up_stimuli.draw()
                VSObject.down_stimuli.draw()
                iframe += 1
                win.flip()
            # initialise index position
            index = int(trial['index'])
            if index == 0:
                left_tex = VSObject.left_index
                right_tex = VSObject.right_stimuli
                up_tex = VSObject.up_stimuli
                down_tex = VSObject.down_stimuli
            elif index == 1:
                left_tex = VSObject.left_stimuli
                right_tex = VSObject.right_index
                up_tex = VSObject.up_stimuli
                down_tex = VSObject.down_stimuli
            elif index == 2:
                left_tex = VSObject.left_stimuli
                right_tex = VSObject.right_stimuli
                up_tex = VSObject.up_index
                down_tex = VSObject.down_stimuli
            elif index == 3:
                left_tex = VSObject.left_stimuli
                right_tex = VSObject.right_stimuli
                up_tex = VSObject.up_stimuli
                down_tex = VSObject.down_index
            # phase II: index(eye shifting)
            iframe = 0
            ref = VSObject.sad_index
            VSObject.cue.play()
            flag = False
            event.clearEvents()
            while iframe < int(fps * index_time):
                # if iframe == 0:
                    # VSObject.win.callOnFlip(port.output_event_data,index + 1)
                # VSObject.girl_index.draw()
                VSObject.hammer_index.draw()
                left_tex.draw()
                right_tex.draw()
                up_tex.draw()
                down_tex.draw()
                keys = event.getKeys(['left', 'right', 'up', 'down'])
                if not flag and bool(keys):
                    flag =True
                    if index == 0 and keys[0] == 'left':
                        ref = VSObject.happy_index
                    elif index == 1 and keys[0] == 'right':
                        ref = VSObject.happy_index
                    elif index == 2 and keys[0] == 'up':
                        ref = VSObject.happy_index
                    elif index == 3 and keys[0] == 'down':
                        ref = VSObject.happy_index
                        
                iframe += 1
                win.flip()
            # phase III: target stimulating
            # port.output_event_data(0)
            # for sf in range(VSObject.stim_frames):
            #     # A
            #         # VSObject.win.callOnFlip(port.output_event_data,index + 1)
            #     # VSObject.girl_index.draw()
            #     VSObject.miner_index.draw()
            #     VSObject.left_stimuli.draw()
            #     VSObject.right_stimuli.draw()
            #     win.flip()
            # phase IV: respond
            iframe = 0
            respond_time = 1
            if ref == VSObject.happy_index:
                VSObject.success.play()
            elif ref == VSObject.sad_index:
                VSObject.fail.play()
            while iframe < int(fps * respond_time):
                ref.draw()
                iframe += 1
                win.flip()
            # phase I: rest state
            if count_trial % 5 == 0:
                rest_time = 5
            if count_trial % 10 == 0:
                rest_time = 10
            if rest_time != 0:
                iframe = 0
                while iframe < int(fps * rest_time):
                    VSObject.gift_index.draw()
                    iframe += 1
                    win.flip()
            rest_time = 1
