import pyvjoy

j = pyvjoy.VJoyDevice(1)

vjoy_max = 32768

j.data.wAxisX = int(0.5 * vjoy_max)
j.data.wAxisY = int(0 * vjoy_max)
j.data.wAxisZ = int(0 * vjoy_max)

j.update()