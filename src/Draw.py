#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""Paint program by Dave Michell.

Subject: tkinter "paint" example
From: Dave Mitchell <davem@magnet.com>
To: python-list@cwi.nl
Date: Fri, 23 Jan 1998 12:18:05 -0500 (EST)

  Not too long ago (last week maybe?) someone posted a request
for an example of a paint program using Tkinter. Try as I might
I can't seem to find it in the archive, so i'll just post mine
here and hope that the person who requested it sees this!

  All this does is put up a canvas and draw a smooth black line
whenever you have the mouse button down, but hopefully it will
be enough to start with.. It would be easy enough to add some
options like other shapes or colors...

                                                yours,
                                                dave mitchell
                                                davem@magnet.com
"""

from Tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cPickle as pickle

"""paint.py: not exactly a paint program.. just a smooth line drawing demo."""

IMAGE_DIMENSIONS = (28, 28)
CANVAS_SCALE = 10

b1 = "up"
x_old, y_old = None, None
drawing_area = None
image = None

def main():
    global IMAGE_DIMENSIONS, CANVAS_SCALE, drawing_area, image
    root = Tk()
    drawing_area = Canvas(root, width=IMAGE_DIMENSIONS[0] * CANVAS_SCALE, height=IMAGE_DIMENSIONS[1] * CANVAS_SCALE, bg="white")
    drawing_area.pack()
    drawing_area.bind("<Motion>", motion)
    drawing_area.bind("<ButtonPress-1>", b1down)
    drawing_area.bind("<ButtonRelease-1>", b1up)
    clear()
    root.mainloop()
    pickle_image = pickle.dumps(image)
    print(pickle_image)

def clear():
    global drawing_area, image
    drawing_area.delete("all")
    image = np.zeros(IMAGE_DIMENSIONS)

def b1down(event):
    global b1
    b1 = "down"           # you only want to draw when the button is down
                          # because "Motion" events happen -all the time-

def b1up(event):
    global b1, x_old, y_old
    b1 = "up"
    xold = None           # reset the line when you let go of the button
    yold = None

def motion(event):
    global CANVAS_SCALE
    if b1 == "down":
        global x_old, y_old
        if x_old is not None and y_old is not None:
            event.widget.create_rectangle(event.x - (CANVAS_SCALE / 2), event.y - (CANVAS_SCALE / 2), event.x + (CANVAS_SCALE / 2), event.y + (CANVAS_SCALE / 2), fill="black")
            image[event.x / CANVAS_SCALE, event.y / CANVAS_SCALE] = 1
        x_old = event.x
        y_old = event.y

if __name__ == "__main__":
    main()
