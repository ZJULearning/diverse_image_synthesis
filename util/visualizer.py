import numpy as np
import os
import ntpath
import time
from . import other
from . import html

class Visualizer():
    def __init__(self, args):
        # self.opt = opt
        self.displayId = 1
        self.useHTML = True
        self.windowSize = 256
        self.name = args.taskname
        if self.displayId > 0:
            import visdom
            self.vis = visdom.Visdom(port = 8097)
            self.numColumns = 0

        if self.useHTML:
            self.webDirectory = os.path.join("./checkpoint", args.taskname, 'web')
            self.imageDirectory = os.path.join(self.webDirectory, 'images')
            print('create web directory %s...' % self.webDirectory)
            other.mkdirs([self.webDirectory, self.imageDirectory])
        self.logName = os.path.join("./checkpoint", args.taskname, 'loss_log.txt')
        with open(self.logName, "a") as logFile:
            now = time.strftime("%c")
            logFile.write('================ Training Loss (%s) ================\n' % now)

        self.startTime = time.time()

    def display_current_results(self, visuals, epoch):
        if self.displayId > 0:
            if self.numColumns > 0:
                ncols = self.numColumns
                title = self.name
                labelHTML = ''
                labelHTMLRow = ''
                nrows = int(np.ceil(len(visuals.items()) / ncols))
                images = []
                idx = 0
                for label, imageNumpy in visuals.items():
                    labelHTMLRow += '<td>%s</td>' % label
                    images.append(imageNumpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        labelHTML += '<tr>%s</tr>' % labelHTMLRow
                        labelHTMLRow = ''
                while idx % ncols != 0:
                    whiteImage = np.ones_like(imageNumpy.transpose([2, 0, 1]))*255
                    images.append(whiteImage)
                    labelHTMLRow += '<td></td>'
                    idx += 1
                if labelHTMLRow != '':
                    labelHTML += '<tr>%s</tr>' % labelHTMLRow
                self.vis.images(images, nrow=ncols, win=self.displayId + 1,
                              opts=dict(title=title + ' images')) # pane col = image row
                labelHTML = '<table style="border-collapse:separate;border-spacing:10px;">%s</table' % labelHTML
                self.vis.text(labelHTML, win = self.displayId + 2,
                              opts=dict(title=title + ' labels'))
            else:
                idx = 1
                for label, imageNumpy in visuals.items():
                    #imageNumpy = np.flipud(imageNumpy)
                    if len(imageNumpy.shape) == 3:
                        self.vis.image(imageNumpy.transpose([2,0,1]), opts=dict(title=label),
                                       win=self.displayId + idx)
                    else:
                        self.vis.image(imageNumpy, opts=dict(title=label),
                                       win=self.displayId + idx)
                    idx += 1

        if self.useHTML:
            for label, imageNumpy in visuals.items():
                imagePath = os.path.join(self.imageDirectory, 'epoch%.3d_%s.png' % (epoch, label))
                other.save_image(imageNumpy, imagePath)
            webpage = html.HTML(self.webDirectory, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, imageNumpy in visuals.items():
                    imagePath = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(imagePath)
                    txts.append(label)
                    links.append(imagePath)
                webpage.add_images(ims, txts, links, width=self.windowSize)
            webpage.save()

    def plot_current_errors(self, epoch, counterRatio, errors):
        if not hasattr(self, 'plotData'):
            self.plotData = {'X':[],'Y':[], 'legend':list(errors.keys())}
        self.plotData['X'].append(epoch + counterRatio)
        self.plotData['Y'].append([errors[k] for k in self.plotData['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plotData['X'])]*len(self.plotData['legend']),1),
            Y=np.array(self.plotData['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plotData['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.displayId)

    def print_current_errors(self, epoch, i, errors):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, time.time() - self.startTime)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.logName, "a") as logFile:
            logFile.write('%s\n' % message)

    def save_images(self, webpage, visuals, name):
        imageDirectory = webpage.get_image_dir()

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, imageNumpy in visuals.items():
            imageName = '%s_%s.png' % (name, label)
            savePath = os.path.join(imageDirectory, imageName)
            other.save_image(imageNumpy, savePath)

            ims.append(imageName)
            txts.append(label)
            links.append(imageName)
        webpage.add_images(ims, txts, links, width=self.windowSize)
