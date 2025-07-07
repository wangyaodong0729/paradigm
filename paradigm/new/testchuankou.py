# 方式1：调用函数接口打开串口时传入配置参数
import serial

# 打开 COM17，将波特率配置为115200.
ser = serial.Serial(port="COM3", baudrate=115200)

# 串口发送 ABCDEFG，并输出发送的字节数。
write_len = ser.write("ABCDEFG".encode('utf-8'))
print("串口发出{}个字节。".format(write_len))

ser.close()
