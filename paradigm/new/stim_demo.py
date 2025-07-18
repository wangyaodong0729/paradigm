#!/usr/bin/env python
# -*- coding: utf-8 -*-


from psychopy import monitors
import numpy as np
#
# 
# 
# from metabci.brainstim.paradigm import SSVEP, MI, paradigm
from paradigm import MyStim1, MyStim2, paradigm


from metabci.brainstim.framework import Experiment


if __name__ == '__main__':
    mon = monitors.Monitor(
        name='1',
        width=59.6, distance=60,  # width 显示器尺寸cm; distance 受试者与显示器间的距离
        verbose=False
    )
    mon.setSizePix([1280, 800])  # 显示器的分辨率
    # mon.save()  # 保存mon后，在实验电脑上会出问题
    bg_color_warm = np.array([0, 0, 0])
    win_size = np.array([1280, 800])
    # esc/q退出开始选择界面
    ex = Experiment(
        monitor=mon,
        bg_color_warm=bg_color_warm,  # 范式选择界面背景颜色[-1~1,-1~1,-1~1]
        screen_id=0,
        win_size=win_size,  # 范式边框大小(像素表示)，默认[1920,1080]
        is_fullscr=True,  # True全窗口,此时win_size参数默认屏幕分辨率
        record_frames=False,
        disable_gc=False,
        process_priority='normal',
        use_fbo=False
    )
    win = ex.get_window()

    # q退出范式界面
    # '''
    # SSVEP
    # '''
    # n_elements, rows, columns = 20, 4, 5  # n_elements 指令数量;  rows 行;  columns 列
    # stim_length, stim_width = 150, 150  # ssvep单指令的尺寸
    # stim_color, tex_color = [1, 1, 1], [1, 1, 1]  # 指令的颜色，文字的颜色
    # fps = 120  # 屏幕刷新率
    # stim_time = 1  # 刺激时长
    # stim_opacities = 1  # 刺激对比度
    # freqs = np.arange(8, 16, 0.4)  # 指令的频率
    # phases = np.array([i * 0.35 % 2 for i in range(n_elements)])  # 指令的相位

    # basic_ssvep = SSVEP(win=win)

    # basic_ssvep.config_pos(n_elements=n_elements, rows=rows, columns=columns, stim_length=stim_length,
    #                        stim_width=stim_width)
    # basic_ssvep.config_text(tex_color=tex_color)
    # basic_ssvep.config_color(refresh_rate=fps, stim_time=stim_time, stimtype='sinusoid', stim_color=stim_color,
    #                          stim_opacities=stim_opacities, freqs=freqs, phases=phases)
    # basic_ssvep.config_index()
    # basic_ssvep.config_response()

    # bg_color = np.array([-1, -1, -1])  # 背景颜色
    # display_time = 1  # 范式开始1s的warm时长
    # index_time = 0.5  # 提示时长，转移视线
    # rest_time = 0.5  # 提示后的休息时长

    # response_time = 1  # 在线反馈
    # port_addr = 0x5efc                                  # 采集主机端口
    # nrep = 1  # block数目
    # lsl_source_id = 'meta_online_worker'  # None                 # source id
    # online = False  # True                                       # 在线实验的标志
    # ex.register_paradigm('basic SSVEP', paradigm, VSObject=basic_ssvep, bg_color=bg_color, display_time=display_time,
    #                      index_time=index_time, rest_time=rest_time, response_time=response_time, port_addr=port_addr,
    #                      nrep=nrep,
    #                      pdim='ssvep', lsl_source_id=lsl_source_id, online=online)

    # '''
    # MI
    # '''
    # fps = 120  # 屏幕刷新率
    # text_pos = (0.0, 0.0)  # 提示文本位置
    # left_pos = [[-480, 0.0]]  # 左手位置
    # right_pos = [[480, 0.0]]  # 右手位置
    # tex_color = 2 * np.array([179, 45, 0]) / 255 - 1  # 提示文本颜色
    # normal_color = [[-0.8, -0.8, -0.8]]  # 默认颜色
    # image_color = [[1, 1, 1]]  # 提示或开始想象颜色
    # symbol_height = 100  # 提示文本的高度
    # n_Elements = 1  # 左右手各一个
    # stim_length = 288  # 长度
    # stim_width = 288  # 宽度
    # basic_MI = MI(win=win)
    # basic_MI.config_color(refresh_rate=fps, text_pos=text_pos, left_pos=left_pos, right_pos=right_pos,
    #                       tex_color=tex_color,
    #                       normal_color=normal_color, image_color=image_color, symbol_height=symbol_height,
    #                       n_Elements=n_Elements,
    #                       stim_length=stim_length, stim_width=stim_width)
    # basic_MI.config_response()
    #
    # bg_color = np.array([-1, -1, -1])  # 背景颜色
    # display_time = 1  # 范式开始1s的warm时长
    # index_time = 1  # 提示时长，转移视线
    # rest_time = 1  # 提示后的休息时长
    # image_time = 4  # 想象时长
    # response_time = 2  # 在线反馈
    # port_addr = "COM3"                                  # 采集主机端口
    # nrep = 10  # block数目
    # lsl_source_id = 'meta_online_worker'  # source id
    # online = False  # True                                       # 在线实验的标志
    # ex.register_paradigm('basic MI', paradigm, VSObject=basic_MI, bg_color=bg_color, display_time=display_time,
    #                      index_time=index_time,
    #                      rest_time=rest_time, response_time=response_time, port_addr=port_addr, nrep=nrep,
    #                      image_time=image_time,
    #                      pdim='mi', lsl_source_id=lsl_source_id, online=online)

    '''
    myStim1
    '''
    fps = 60  # 屏幕刷新率
    text_pos = (0.0, 0.0)  # 提示文本位置
    left_pos = [[-480, 0.0]]  # 左手位置
    right_pos = [[480, 0.0]]  # 右手位置
    tex_color = 2 * np.array([179, 45, 0]) / 255 - 1  # 提示文本颜色
    normal_color = [[-0.8, -0.8, -0.8]]  # 默认颜色
    image_color = [[2, 2, 2]]  # 提示或开始想象颜色
    flash_color = [2, 2, 2]
    symbol_height = 100  # 提示文本的高度
    n_Elements = 1  # 左右手各一个
    stim_length = 288  # 长度
    stim_width = 288  # 宽度
    stim_time = 2
    freqs = np.array([7.5, 8])  # 指令的频率
    phases = np.array([0, 0.35])

    hybrid_bci = MyStim1(win=win)

    hybrid_bci.config_stim()
    hybrid_bci.config_flash(refresh_rate=fps, stim_time=stim_time, flash_color=flash_color, freqs=freqs, phases=phases)
    hybrid_bci.config_response()

    bg_color = np.array([-1, -1, -1])  # 背景颜色
    display_time = 1  # 范式开始1s的warm时长
    index_time = 3  # 提示时长，转移视线
    rest_time = 1  # 提示后的休息时长
    response_time = 1  # 在线反馈
    nrep = 10  # block数目
    
    # port_addr = "COM3"                                  # 采集主机端口
    # online = True
    # lsl_source_id = None  # 'meta_online_worker'  # None                 # source id
    
    port_addr = None
    lsl_source_id = "meta_online_worker"
    online = False

    ex.register_paradigm('hybrid BCI1', paradigm,
                         VSObject=hybrid_bci,
                         bg_color=bg_color,
                         display_time=display_time, index_time=index_time,
                         rest_time=rest_time,
                         port_addr=port_addr,
                         nrep=nrep,
                         pdim='hybrid_bci1', lsl_source_id=lsl_source_id,
                         online=online)
    

    '''
    myStim2
    '''
    fps = 60  # 屏幕刷新率
    text_pos = (0.0, 0.0)  # 提示文本位置
    left_pos = [[-480, 0.0]]  # 左手位置
    right_pos = [[480, 0.0]]  # 右手位置
    tex_color = 2 * np.array([179, 45, 0]) / 255 - 1  # 提示文本颜色
    normal_color = [[-0.8, -0.8, -0.8]]  # 默认颜色
    image_color = [[2, 2, 2]]  # 提示或开始想象颜色
    flash_color = [2, 2, 2]
    symbol_height = 100  # 提示文本的高度
    n_Elements = 1  # 左右手各一个
    stim_length = 288  # 长度
    stim_width = 288  # 宽度
    stim_time = 2
    freqs = np.array([7.5, 8])  # 指令的频率
    phases = np.array([0, 0.35])

    hybrid_bci = MyStim2(win=win)

    hybrid_bci.config_stim()
    hybrid_bci.config_flash(refresh_rate=fps, stim_time=stim_time, flash_color=flash_color, freqs=freqs, phases=phases)
    hybrid_bci.config_response()

    bg_color = np.array([-1, -1, -1])  # 背景颜色
    display_time = 1  # 范式开始1s的warm时长
    index_time = 3  # 提示时长，转移视线
    rest_time = 1  # 提示后的休息时长
    response_time = 1  # 在线反馈
    nrep = 10  # block数目
    
    # port_addr = "COM3"                                  # 采集主机端口
    # online = True
    # lsl_source_id = None  # 'meta_online_worker'  # None                 # source id
    
    port_addr = None
    lsl_source_id = "meta_online_worker"
    online = False

    ex.register_paradigm('hybrid BCI2', paradigm,
                         VSObject=hybrid_bci,
                         bg_color=bg_color,
                         display_time=display_time, index_time=index_time,
                         rest_time=rest_time,
                         port_addr=port_addr,
                         nrep=nrep,
                         pdim='hybrid_bci2', lsl_source_id=lsl_source_id,
                         online=online)

    ex.run()

