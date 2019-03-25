import dominate
from dominate.tags import *
import os


class HTML:
    def __init__(self, webDirectory, title, reflesh=0):
        self.title = title
        self.webDirectory = webDirectory
        self.imageDirectory = os.path.join(self.webDirectory, 'images')
        if not os.path.exists(self.webDirectory):
            os.makedirs(self.webDirectory)
        if not os.path.exists(self.imageDirectory):
            os.makedirs(self.imageDirectory)

        self.doc = dominate.document(title=title)
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh))

    def get_image_dir(self):
        return self.imageDirectory

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, links, width=400):
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style="width:%dpx" % width, src=os.path.join('images', im))
                            br()
                            p(txt)

    def save(self):
        htmlFile = '%s/index.html' % self.webDirectory
        f = open(htmlFile, 'wt')
        f.write(self.doc.render())
        f.close()
