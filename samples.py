"""Input/Output code to read in the classification data.

Champlain College CSI-480, Fall 2018
The following code was adapted by Joshua Auerbach (jauerbach@champlain.edu)
from the UC Berkeley Pacman Projects (see license and attribution below).

----------------------
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).
"""


import util
import zipfile
import os
import pickle


# Module Classes

class Datum:
    """A datum is a pixel-level encoding of digits or face/non-face edge maps.

    Digits are from the MNIST dataset and face images are from the
    easy-faces and background categories of the Caltech 101 dataset.

    Each digit is 28x28 pixels, and each face/non-face image is 60x74
    pixels, each pixel can take the following values:
      0: no edge (blank)
      1: gray pixel (+) [used for digits only]
      2: edge [for face] or black pixel [for digit] (#)

    Pixel data is stored in the 2-dimensional array pixels, which
    maps to pixels on a plane according to standard euclidean axes
    with the first dimension denoting the horizontal and the second
    the vertical coordinate:

      28 # # # #      #  #
      27 # # # #      #  #
       .
       .
       .
       3 # # + #      #  #
       2 # # # #      #  #
       1 # # # #      #  #
       0 # # # #      #  #
         0 1 2 3 ... 27 28

    For example, the + in the above diagram is stored in pixels[2][3], or
    more generally pixels[column][row].

    The contents of the representation can be accessed directly
    via the get_pixel and get_pixels methods.
    """

    def __init__(self, data, width, height):
        """Create a new datum from file input (standard MNIST encoding)."""
        self.height = height
        self.width = width
        if data is None:
            data = [[' ' for i in range(width)] for j in range(height)]
        self.pixels = util.array_invert(convert_to_integer(data))

    def get_pixel(self, column, row):
        """Return the value of the pixel at column, row as 0, or 1."""
        return self.pixels[column][row]

    def get_pixels(self):
        """Return all pixels as a list of lists."""
        return self.pixels

    def get_ascii_string(self):
        """Render the data item as an ascii image."""
        rows = []
        data = util.array_invert(self.pixels)
        for row in data:
            ascii = list(map(integer_to_ascii_symbol, row))
            rows.append("".join(ascii))
        return "\n".join(rows)

    def __str__(self):
        """Return data item as an ascii image."""
        return self.get_ascii_string()


# Data processing, cleanup and display functions

def load_data_file(filename, n, width, height):
    """Read n data images from a file and returns a list of Datum objects.

    (Return less then n items if the end of file is encountered).
    """
    fin = readlines(filename)
    fin.reverse()
    items = []
    for i in range(n):
        data = []
        for j in range(height):
            data.append(list(fin.pop()))
        if len(data[0]) < width-1:
            # we encountered end of file...
            print("Truncating at %d examples (maximum)" % i)
            break
        items.append(Datum(data, width, height))
    return items


def readlines(filename):
    """Open a file or read it from the zip archive data.zip."""
    if(os.path.exists(filename)):
        return [l[:-1] for l in open(filename).readlines()]
    else:
        z = zipfile.ZipFile('data.zip')
        return str(z.read(filename)).split('\n')


def load_labels_file(filename, n):
    """Read n labels from a file and return a list of integers."""
    fin = readlines(filename)
    labels = []
    for line in fin[:min(n, len(fin))]:
        if line == '':
            break
        labels.append(int(line))
    return labels


def load_pacman_states_file(filename):
    """Read pacman states from file."""
    f = open(filename, 'rb')
    result = pickle.load(f)
    f.close()
    return result


def load_pacman_data(filename, n):
    """Return (inputs, labels) from specified recorded games.

    inputs are game states
    labels are actions
    """
    components = load_pacman_states_file(filename)
    return components['states'][:n], components['actions'][:n]


def integer_to_ascii_symbol(value):
    """Convert value to ascii symbol."""
    if(value == 0):
        return ' '
    elif(value == 1):
        return '+'
    elif(value == 2):
        return '#'


def ascii_symbol_to_integer(character):
    """Convert ascii symbol to integer."""
    if(character == ' '):
        return 0
    elif(character == '+'):
        return 1
    elif(character == '#'):
        return 2


def convert_to_integer(data):
    """Convert possible many symbols to integers."""
    if not isinstance(data, list):
        return ascii_symbol_to_integer(data)
    else:
        return list(map(convert_to_integer, data))


# Testing

def _test():
    import doctest
    doctest.testmod()  # Test the interactive sessions in function comments
    n = 1
    #  items = load_data_file("facedata/facedatatrain", n,60,70)
    #  labels = load_labels_file("facedata/facedatatrainlabels", n)
    items = load_data_file("digitdata/trainingimages", n, 28, 28)
    # labels = load_labels_file("digitdata/traininglabels", n)
    for i in range(1):
        print(items[i])
        print(items[i])
        print((items[i].height))
        print((items[i].width))
        print(dir(items[i]))
        print(items[i].get_pixels())


if __name__ == "__main__":
    _test()
