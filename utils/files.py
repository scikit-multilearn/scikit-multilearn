import os, shutil
from itertools import ifilter

class files(object):
    @staticmethod
    def emptyDir(directory):
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

    @staticmethod
    def removeDir(directory):
        if os.path.isdir(directory):
            print 'removing : %s' % directory
            shutil.rmtree(directory)

    @staticmethod
    def removeDirsRecursive(root, filterExpr):
        for root, dirs, _ in os.walk(root):
            for directory in ifilter(filterExpr, dirs):
                dirPath = os.path.join(root, directory)
                shutil.rmtree(dirPath)

    @staticmethod
    def removeFilesRecursive(root, filterExpr):
        for root, _, files in os.walk(root):
            for fileName in ifilter(filterExpr, files):
                filePath = os.path.join(root, fileName)
                os.remove(filePath)

    @staticmethod
    def actionOnFilesRecursive(root, filterExpr, action):
        for root, _, files in os.walk(root):
            for fileName in ifilter(filterExpr, files):
                action(root, fileName)
