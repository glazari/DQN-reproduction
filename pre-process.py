import numpy as np

#INPUT ( frames)
#the frames are a m-tuple of the m previous frame
max_frames = np.maximum.reduce(obs)
xyz = color.rgb2xyz(max_frames)
y = xyz[:,:,1]	
small = resize(y,(84,84))	

