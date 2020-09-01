import cv2
from PIL import Image, ImageStat

# This movie
def detect_color_image(file, thumb_size=40, MSE_cutoff=22, adjust_color_bias=True):
    pil_img = file
    bands = pil_img.getbands()
    if bands == ('R', 'G', 'B') or bands== ('R', 'G', 'B', 'A'):
        thumb = pil_img.resize((thumb_size, thumb_size))
        SSE, bias = 0, [0,0,0]
        if adjust_color_bias:
            bias = ImageStat.Stat(thumb).mean[:3]
            bias = [b - sum(bias)/3 for b in bias ]
        for pixel in thumb.getdata():
            mu = sum(pixel)/3
            SSE += sum((pixel[i] - mu - bias[i])*(pixel[i] - mu - bias[i]) for i in [0,1,2])
        MSE = float(SSE)/(thumb_size*thumb_size)
        if MSE <= MSE_cutoff:
            return False
        else:
            return True
    elif len(bands)==1:
        return False


vidcap = cv2.VideoCapture('Loving.Vincent.2017.1080p.BluRay.x264-[YTS.AG].mp4')  # open the video file
success, image = vidcap.read()
count = 0
while success:
    if count % 2 == 0: # this movie is painted and so every painting is displayed twice
        if detect_color_image(Image.fromarray(image.astype('uint8'), 'RGB')):
            cv2.imwrite("./frames/colour/frame%d.jpg" % count, image)  # save in colour if colour
        else:
            cv2.imwrite("./frames/bw/frame%d.jpg" % count, image)  # save in bw if black and white
    success, image = vidcap.read()  # read another frame
    print('Read a new frame: ', success)
    count += 1
