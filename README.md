# SHVideoSort
 
Video sorting algorithm based upon motion

1) Run background subtraction to give an 8-bit foreground mask showing motion at each frame
2) Sum pixels over foreground mask to provide an indicator of the level of motion (motion metric)
3) For each new maximum value of the motion metric, run blob detection on foreground mask filtering out blobs smaller than a fixed number of pixels but saving the others
4) At the end, a video is stored as interesting if motion AND blob with maximum size are greater than set thresholds

The frame and mask with maximum motion is saves as a PNG and  some details about the video are recorded to CSV file e.g. filename, duration, motion metric max, maximum blob size, time of maximum motion in video

Issues:
1) The above assumes the largest moving object in the video is of most interest
2) Needs to employ CUDA
